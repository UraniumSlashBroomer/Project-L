## Extended abstract of this project / короткая статья об этом проекте

Ru: Задача, архитектура модели и ключевые метрики этого проекта описаны в краткой статье на 2 страницы. Статья на английском языке, так как я её делал для летней школы EEML. Полный документ загружен в репозиторий: **[Прочитать полностью (2 страницы, PDF)](extended_abstract.pdf)**

---

Eng: The core methodology, framework, and key metrics of this project are documented in a short 2-page extended abstract. Full document uploaded to this repository: **[Read Extended Abstract (2 Pages, PDF)](extended_abstract.pdf)**

---

<img width="799" height="829" alt="image" src="https://github.com/user-attachments/assets/e3b671a2-ad15-404c-9812-2de45286dc8e" />

# Project-L

## Language
- [Русский](#russian)

- [English](#english)

---
## Russian:
**Суть проекта - распознавать движение руки в виде трясущейся буквы L.** Данный проект был реализован для применения методов компьютерного зрения и ознакомления с **STGCN (spatio-temporal graph convolutional network)**. Сам результат не несет практической пользы. Распозновать трясущуюся букву L было выбрано целью _из-за простоты создания набора данных_ (он был создан вручную) и _просто ради забавы_.

---

### Используемый пайплайн и файлы проекта

В проекте используется следующий пайплайн:

1. mediapipe (модель для распознавания графа руки) достает нормализованные координаты
2. полученные точки сохраняются в буфер _длиною в 30 кадров_
3. каждые _5 кадров_ буфер отправляется на инференс в обученный STGCN.

В данном проекте есть следующие файлы отвечающие каждый за свое:

- **camera.py** - файл для записи датасета
- **STGCN.py** - файл с реализацией STGCN
- **main.ipynb** - этот ноутбук для обучения и оценки модели
- **model_inference.py** - файл для инференса модели
- **.pth файлы** - сохраненные модели и их результаты
- **nums.txt** - файл для сбора датасета

---

### Установка
1. Клонируем репозиторий и переходим в папку с проектом:

```
git clone https://github.com/UraniumSlashBroomer/Project-L.git
cd Project-L
```

2. Создаем виртуальную среду, активируем и устанавливаем зависимости:
```
python -m venv .env
source .env/bin/activate
pip install -r requirements.txt
```

**ВАЖНО!**
В requirements находится torch без поддержки GPU. **Если нужно обучить модель на GPU - установите torch с поддержкой GPU (CUDA 12.6)** вручную следующей командой:
```
pip install torch torchvision --index-url https://download.pytorch.org/whl/cu126
```

---

### Запуск модели
Инференс модели находится в файле model_inference.py. 

**Чтобы закрыть окно с камерой нажмите ESC**

Каждые 5 кадров в консоль на данный момент выводится следущее:

_(вероятность принадлежности к классу 0, вероятность принадлежности к классу 1)_

_итоговый ответ модели_

- [Пример](#example)
---
## English:

**Project goal — recognizing a hand movement shaped like a shaky “L”.**
This project was implemented to apply computer vision methods and to get familiar with **STGCN (spatio-temporal graph convolutional network)**. The result itself has no practical use. Detecting a shaky “L” was chosen as the goal due to the _simplicity of creating the dataset_ (it was created manually) and just for fun.

---

### Pipeline and project files

The project uses the following pipeline:

- mediapipe (a model for hand graph detection) extracts normalized coordinates
- the obtained points are stored in a buffer of length 30 frames
- every 5 frames, the buffer is sent for inference to a trained STGCN

The project includes the following files, each responsible for its own part:

- **camera.py** — dataset recording script
- **STGCN.py** — STGCN implementation
- **main.ipynb** — notebook for training and evaluating the model
- **model_inference.py** — model inference script
- **.pth files** — saved models and their results
- **nums.txt** — file for collecting the dataset

---

### Installation
Clone the repository and navigate to the project directory:
```
git clone https://github.com/UraniumSlashBroomer/Project-L.git
cd Project-L
```
Create a virtual environment, activate it, and install dependencies:
```
python -m venv .env
source .env/bin/activate
pip install -r requirements.txt
```

**IMPORTANT!**
The requirements include a CPU-only version of torch. **If you need to train the model on a GPU, install a GPU-enabled version of torch (CUDA 12.6) manually** using the following command:

```
pip install torch torchvision --index-url https://download.pytorch.org/whl/cu126
```

---

### Running the model

Model inference is implemented in the file model_inference.py, just run this file.

**Press ESC to close the camera window**

Every 5 frames, the following is currently printed to the console:

__(probability of belonging to class 0, probability of belonging to class 1)__

__final (class) model prediction__

---

### Example:
---
![2025-10-03 20-52-06_1](https://github.com/user-attachments/assets/6a083634-13d6-4db6-a8aa-3699d950e181)

