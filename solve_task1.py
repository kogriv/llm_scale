import pandas as pd
import numpy as np
from sklearn.ensemble import RandomForestRegressor, GradientBoostingRegressor
from sklearn.model_selection import cross_val_score
from sklearn.preprocessing import PolynomialFeatures
import json

# Загрузка данных
train_data = pd.read_csv('data/task1/train_weights.csv')
test_data = pd.read_csv('data/task1/test_weights.csv')

# Разделение на признаки и целевую переменную
X_train = train_data.drop('MSE', axis=1)
y_train = train_data['MSE']
X_test = test_data

print(f"Размер обучающей выборки: {X_train.shape}")
print(f"Размер тестовой выборки: {X_test.shape}")
print(f"Статистика целевой переменной MSE:")
print(y_train.describe())

# Попробуем несколько моделей
# Модель 1: Random Forest
rf_model = RandomForestRegressor(n_estimators=200, max_depth=20, random_state=42, min_samples_split=5)
rf_model.fit(X_train, y_train)
rf_pred = rf_model.predict(X_test)

# Модель 2: Gradient Boosting
gb_model = GradientBoostingRegressor(n_estimators=200, max_depth=5, learning_rate=0.1, random_state=42)
gb_model.fit(X_train, y_train)
gb_pred = gb_model.predict(X_test)

# Модель 3: Полиномиальные признаки + Gradient Boosting
poly = PolynomialFeatures(degree=2, include_bias=False)
X_train_poly = poly.fit_transform(X_train)
X_test_poly = poly.transform(X_test)

gb_poly_model = GradientBoostingRegressor(n_estimators=200, max_depth=5, learning_rate=0.1, random_state=42)
gb_poly_model.fit(X_train_poly, y_train)
gb_poly_pred = gb_poly_model.predict(X_test_poly)

# Ансамбль моделей (усреднение предсказаний)
ensemble_pred = (rf_pred + gb_pred + gb_poly_pred) / 3

# Оценка моделей с помощью кросс-валидации
print("\nОценка моделей (R²):")
rf_score = cross_val_score(rf_model, X_train, y_train, cv=5, scoring='r2').mean()
print(f"Random Forest: {rf_score:.4f}")

gb_score = cross_val_score(gb_model, X_train, y_train, cv=5, scoring='r2').mean()
print(f"Gradient Boosting: {gb_score:.4f}")

# Используем ансамбль для финального предсказания
final_predictions = ensemble_pred

# Формирование результата в требуемом формате
result = []
for i in range(len(test_data)):
    entry = {}
    for col in test_data.columns:
        entry[col] = float(test_data[col].iloc[i])
    entry['MSE'] = round(float(final_predictions[i]), 3)
    result.append(entry)

# Сохранение в JSON
with open('answers', 'w') as f:
    json.dump(result, f, indent=4)

print(f"\nПредсказания сохранены в файл 'answers'")
print(f"Пример первых 3 предсказаний:")
for i in range(min(3, len(result))):
    print(f"  Образец {i+1}: MSE = {result[i]['MSE']}")
