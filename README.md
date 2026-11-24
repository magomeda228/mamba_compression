# Mamba-2 + n-gram compressor

Python 3.10+. Входной файл `data/enwik5` включён.

## Установка и запуск

```bash
python -m pip install -r requirements.txt
python main.py
python verify.py data/enwik5 data/enwik5_decompressed.dat
```

`main.py` сначала целиком запускает исходную LSTM без тюнинга, затем нашу
реализацию на том же устройстве. Обе распаковывают файл и проверяют каждый
байт. В конце выводятся размеры архивов, коэффициенты и время, сравнение
сохраняется в `results.json`.

LSTM создаёт `data/enwik5_lstm_compressed.dat` и
`data/enwik5_lstm_decompressed.dat`. Наша версия создаёт
`data/enwik5_compressed.dat` и `data/enwik5_decompressed.dat`.

## Дополнительные команды

```bash
python main.py --device cpu
python main.py --profile quality
python main.py --only-ours
python main.py --mode compress
python main.py --mode decompress
python main.py --input data/my_file.bin
```

По умолчанию: K=7, alpha=4, batch=288. Профиль `quality`: K=10, alpha=4,
batch=48. Точные настройки находятся в `profiles.json`.

Устройство выбирается автоматически. Для GPU нужна совместимая CUDA-сборка
PyTorch. На Linux при наличии Triton и C++-компилятора включается ускорение;
C++-библиотека собирается при первом запуске. Без него используются
Python-таблицы. Первый запуск дольше из-за компиляции. Архивы разных
вычислительных backend требуют соответствующего backend при распаковке.

Все Python-модули в этой папке участвуют в запуске LSTM, нашего кодека или
его CUDA-ускорения. `fast_memory.cpp` нужен для ускоренной статистики на CPU.

## Исходники

Базовая LSTM `lstm_codec.py` сохранён без изменений. Mamba-2 реализована в `mamba2.py`.

Copyright © 2026 Project Nayuki. (MIT License)

Permission is hereby granted, free of charge, to any person obtaining a copy of
this software and associated documentation files (the "Software"), to deal in
the Software without restriction, including without limitation the rights to
use, copy, modify, merge, publish, distribute, sublicense, and/or sell copies of
the Software, and to permit persons to whom the Software is furnished to do so,
subject to the following conditions:

* The above copyright notice and this permission notice shall be included in
  all copies or substantial portions of the Software.

* The Software is provided "as is", without warranty of any kind, express or
  implied, including but not limited to the warranties of merchantability,
  fitness for a particular purpose and noninfringement. In no event shall the
  authors or copyright holders be liable for any claim, damages or other
  liability, whether in an action of contract, tort or otherwise, arising from,
  out of or in connection with the Software or the use or other dealings in the
  Software.

Source: https://github.com/nayuki/Reference-arithmetic-coding

# mamba_compression
