// CPU port of context_memory.py. Compile without fast-math/FMA contraction:
// encoder and decoder must reproduce the same integer frequency tables.
#include <pybind11/numpy.h>
#include <pybind11/pybind11.h>

#include <algorithm>
#include <cmath>
#include <cstdint>
#include <limits>
#include <string>
#include <unordered_map>
#include <vector>

namespace py = pybind11;

namespace {
// Исторические пороги исходной модели; отдельно они не подбирались.
// Деление пополам постепенно уменьшает вес старых наблюдений.
constexpr int kGlobalRescaleThreshold = 32768;
constexpr int kContextRescaleThreshold = 16384;
constexpr double kFrequencyScale = 16000000.0;

struct ContextEntry {
    std::unordered_map<int, int> counts;
    int total = 0;
};

class ContinuationMemory {
  public:
    ContinuationMemory(int vocab_size, int batch_size, int order, double initial_weight,
                       double rate, double escape_scale)
        : vocab_size_(vocab_size), order_(order), global_total_(vocab_size), mixture_rate_(rate),
          escape_scale_(escape_scale) {
        if (vocab_size < 1 || vocab_size > 256 || batch_size < 1 || order < 0 || order > 220) {
            throw py::value_error(
                "Expected vocab_size in [1, 256], positive batch_size, order in [0, 220]");
        }
        if (!std::isfinite(initial_weight) || initial_weight <= 0 || initial_weight >= 1 ||
            !std::isfinite(rate) || rate < 0 || !std::isfinite(escape_scale) || escape_scale < 0) {
            throw py::value_error("Invalid mixture weight, learning rate, or escape scale");
        }
        log_odds_ = std::log(initial_weight / (1 - initial_weight));
        global_counts_.assign(vocab_size, 1.0);
        histories_.resize(batch_size);
    }

    double weight() const {
        return 1 / (1 + std::exp(-log_odds_));
    }

    py::array_t<double> predict(int stream) const {
        const auto& history = histories_.at(stream);
        py::array_t<double> result(vocab_size_);
        auto* probabilities = result.mutable_data();
        for (int symbol = 0; symbol < vocab_size_; ++symbol) {
            probabilities[symbol] = global_counts_[symbol] / global_total_;
        }
        // P_k uses P_{k-1}; an unknown context keeps the shorter forecast intact.
        for (int length = 1; length <= std::min(order_, static_cast<int>(history.size()));
             ++length) {
            const auto found = tables_.find(history.substr(history.size() - length));
            if (found == tables_.end()) {
                continue;
            }
            const auto& entry = found->second;
            const double escape = escape_scale_ * entry.counts.size();
            const double divisor = entry.total + escape;
            const double factor = escape / divisor;
            for (int symbol = 0; symbol < vocab_size_; ++symbol) {
                probabilities[symbol] *= factor;
            }
            for (const auto& [symbol, count] : entry.counts) {
                probabilities[symbol] += static_cast<double>(count) / divisor;
            }
        }
        return result;
    }

    py::array_t<double> combine(py::array_t<float, py::array::c_style> neural,
                                py::array_t<double, py::array::c_style> memory) const {
        if (neural.ndim() != 1 || memory.ndim() != 1 || neural.size() != vocab_size_ ||
            memory.size() != vocab_size_) {
            throw py::value_error("Each probability vector must match vocab_size");
        }
        py::array_t<double> result(vocab_size_);
        auto* probabilities = result.mutable_data();
        const auto* neural_values = neural.data();
        const auto* memory_values = memory.data();
        const double neural_weight = weight();
        for (int symbol = 0; symbol < vocab_size_; ++symbol) {
            // Match NumPy 2: multiply the NN vector in float32 before adding
            // float64 n-gram probabilities. This order affects archive bits.
            const float neural_term = static_cast<float>(neural_weight) * neural_values[symbol];
            probabilities[symbol] =
                static_cast<double>(neural_term) + (1 - neural_weight) * memory_values[symbol];
        }
        return result;
    }

    void observe(int stream, int symbol, py::object neural_probability,
                 py::object memory_probability) {
        auto& history = histories_.at(stream);
        if (symbol < 0 || symbol >= vocab_size_) {
            throw py::index_error("Symbol is outside the model vocabulary");
        }
        // Обновление только после передачи байта: декодер должен знать тот же символ.
        if (!neural_probability.is_none()) {
            if (memory_probability.is_none()) {
                throw py::value_error("Mixture update requires both expert probabilities");
            }
            const double neural_weight = weight();
            const double neural = neural_probability.cast<double>();
            const double memory = memory_probability.cast<double>();
            const double mixture =
                std::max(neural_weight * neural + (1 - neural_weight) * memory, 1e-12);
            const double gradient =
                neural_weight * (1 - neural_weight) * (neural - memory) / mixture;
            log_odds_ = std::max(-8.0, std::min(8.0, log_odds_ + mixture_rate_ * gradient));
            ++mixture_predictions_;
        }
        ++global_counts_[symbol];
        ++global_total_;
        if (global_total_ >= kGlobalRescaleThreshold) {
            global_total_ = 0;
            for (auto& count : global_counts_) {
                count = std::max(1.0, std::ceil(count / 2));
                global_total_ += count;
            }
        }
        for (int length = 1; length <= std::min(order_, static_cast<int>(history.size()));
             ++length) {
            auto& entry = tables_[history.substr(history.size() - length)];
            ++entry.counts[symbol];
            ++entry.total;
            if (entry.total >= kContextRescaleThreshold) {
                entry.total = 0;
                for (auto& item : entry.counts) {
                    item.second = (item.second + 1) / 2;
                    entry.total += item.second;
                }
            }
        }
        if (order_ > 0) {
            history.push_back(static_cast<char>(symbol));
            if (static_cast<int>(history.size()) > order_) {
                history.erase(0, history.size() - order_);
            }
        }
    }

    py::dict stats() const {
        py::dict result;
        result["context_tables"] = tables_.size();
        result["neural_weight_final"] = weight();
        result["mixture_predictions"] = mixture_predictions_;
        return result;
    }

  private:
    int vocab_size_;
    int order_;
    double global_total_;
    double log_odds_;
    double mixture_rate_;
    double escape_scale_;
    std::uint64_t mixture_predictions_ = 0;
    std::vector<double> global_counts_;
    std::vector<std::string> histories_;
    std::unordered_map<std::string, ContextEntry> tables_;
};

template <typename Float> py::array_t<std::int64_t> cumulative_frequencies(py::array input) {
    // Copy strided NumPy slices before treating their data as a flat pointer.
    const auto contiguous = py::array_t<Float, py::array::c_style | py::array::forcecast>(input);
    const auto* probabilities = contiguous.data();
    py::array_t<std::int64_t> result(contiguous.size());
    auto* frequencies = result.mutable_data();
    std::int64_t total = 0;
    for (py::ssize_t i = 0; i < contiguous.size(); ++i) {
        if (!std::isfinite(probabilities[i]) || probabilities[i] < 0) {
            throw py::value_error("Probabilities must be finite and nonnegative");
        }
        // Keep float32/float64 rounding identical to the Python reference.
        Float count = probabilities[i] * static_cast<Float>(kFrequencyScale);
        count += static_cast<Float>(1.0);
        if (static_cast<double>(count) >=
            static_cast<double>(std::numeric_limits<std::int64_t>::max())) {
            throw py::value_error("Frequency count exceeds int64 capacity");
        }
        const auto integer_count = std::max<std::int64_t>(1, static_cast<std::int64_t>(count));
        if (integer_count > std::numeric_limits<std::int64_t>::max() - total) {
            throw py::value_error("Cumulative frequencies exceed int64 capacity");
        }
        total += integer_count;
        frequencies[i] = total;
    }
    return result;
}

py::array_t<std::int64_t> frequencies(py::array input) {
    if (input.ndim() != 1 || input.size() == 0) {
        throw py::value_error("Expected a nonempty one-dimensional probability vector");
    }
    if (input.dtype().is(py::dtype::of<float>())) {
        return cumulative_frequencies<float>(input);
    }
    return cumulative_frequencies<double>(input);
}
} // namespace

PYBIND11_MODULE(fast_memory, module) {
    py::class_<ContinuationMemory>(module, "ContinuationMemory")
        .def(py::init<int, int, int, double, double, double>(), py::arg("vocab_size"),
             py::arg("batch_size"), py::arg("order") = 6, py::arg("initial_neural_weight") = 0.5,
             py::arg("rate") = 0.01, py::arg("escape_scale") = 1.0)
        .def("predict", &ContinuationMemory::predict)
        .def("combine", &ContinuationMemory::combine)
        .def("weight", &ContinuationMemory::weight)
        .def("observe", &ContinuationMemory::observe, py::arg("stream"), py::arg("symbol"),
             py::arg("neural_probability") = py::none(), py::arg("memory_probability") = py::none())
        .def("stats", &ContinuationMemory::stats);
    module.def("frequencies", &frequencies);
}
