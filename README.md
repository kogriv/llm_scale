# LLM Scale - Решение задач машинного обучения

Этот репозиторий содержит решения задач по машинному обучению из курса LLM Scale.

## Структура проекта

```
llm_scale/
├── data/
│   ├── task1/          # Данные для задачи 1 (регрессия MSE)
│   └── task2/          # Данные для задачи 2 (классификация)
├── solve_task1.py      # Решение задачи 1
├── solve_task2_v3.py   # Лучшее решение задачи 2 (AdaBoost)
├── solve_task2_lgbm.py # Альтернативное решение (LightGBM)
├── task1.md            # Описание задачи 1
├── task2.md            # Описание задачи 2
├── task2_solution_report.md  # Подробный отчет по задаче 2
└── requirements.txt    # Зависимости проекта
```

## Задачи

### Task 1: Предсказание ошибок (Регрессия)
- **Цель**: Предсказать MSE для векторов весов линейной регрессии
- **Метрика**: RMSLE (Root Mean Squared Logarithmic Error)
- **Целевой результат**: RMSLE ≤ 0.2 для максимального балла
- **Решение**: Ансамбль Random Forest, Gradient Boosting и полиномиальные признаки
- **Скрипт**: `solve_task1.py`

### Task 2: Лаборатория биокибернетики (Классификация)
- **Цель**: Классификация биологических объектов (человек/не человек)
- **Метрика**: ROC-AUC
- **Целевой результат**: AUC ≥ 0.88 для максимального балла
- **Лучшее решение**: AdaBoost с ROC-AUC = 0.8681
- **Скрипт**: `solve_task2_v3.py`
- **Отчет**: См. `task2_solution_report.md`

## Установка

### 1. Клонирование репозитория
```bash
git clone https://github.com/kogriv/llm_scale.git
cd llm_scale
```

### 2. Создание виртуального окружения
```bash
python -m venv venv_llm_scale
```

### 3. Активация виртуального окружения

**Windows:**
```bash
venv_llm_scale\Scripts\activate
```

**Linux/Mac:**
```bash
source venv_llm_scale/bin/activate
```

### 4. Установка зависимостей
```bash
pip install -r requirements.txt
```

## Использование

### Задача 1
```bash
python solve_task1.py
```
Результат сохраняется в файл `answers` (JSON формат).

### Задача 2
```bash
python solve_task2_v3.py
```
Результат сохраняется в файл `answers.csv`.

## Основные библиотеки

- **pandas** - работа с данными
- **numpy** - численные вычисления
- **scikit-learn** - классические алгоритмы ML
- **lightgbm** - градиентный бустинг
- **catboost** - градиентный бустинг с поддержкой категориальных признаков
- **xgboost** - экстремальный градиентный бустинг

## Результаты

### Task 1
- Модель: Ансамбль (Random Forest + Gradient Boosting + Polynomial Features)
- Метрика R² на кросс-валидации: ~0.86

### Task 2
- **Лучшая модель**: AdaBoost
- **ROC-AUC на 10-fold CV**: 0.8681
- **Feature Engineering**: 28 признаков (9 исходных + 18 созданных)
- Подробности в `task2_solution_report.md`

## Автор

Репозиторий создан для курса LLM Scale.

## Лицензия

Этот проект предназначен для образовательных целей.
