# object-detection

Проект реализован при помощи Android приложения и библиотеки KotlinDL. 

## Модель
Для использования модели в KotlinDL есть возможность использовать модели <a href="https://github.com/microsoft/onnxruntime-inference-examples">ONNX Runtime </a>.
ONNX предоставляет возможность запускать модели на различные платформах (Unreal Engine/Andorid), поэтому с ним было проще всего реализовать.

Модель сохраняется в файл onnx (в репозитории это файл '''ssdlite_mobilenet_v3.onnx'''). Код сохранения в файл можно посмотреть в ноутбуке <a href="https://github.com/Navu45/object-detection/blob/master/ssd_mobilenet.ipynb">ssd_mobilenet.ipynb</a>

## Android приложение
Результат показан на GIF-изображении ниже.
