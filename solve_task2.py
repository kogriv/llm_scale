import pandas as pd
import numpy as np
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier
from sklearn.model_selection import cross_val_score
from sklearn.preprocessing import LabelEncoder
from sklearn.metrics import roc_auc_score

# Загрузка данных
train_data = pd.read_csv('data/task2/train.csv')
test_data = pd.read_csv('data/task2/test.csv')

print(f"Размер обучающей выборки: {train_data.shape}")
print(f"Размер тестовой выборки: {test_data.shape}")
print(f"\nПропущенные значения в train:")
print(train_data.isnull().sum())
print(f"\nПропущенные значения в test:")
print(test_data.isnull().sum())
print(f"\nРаспределение классов:")
print(train_data['target'].value_counts())

# Разделение на признаки и целевую переменную
X_train = train_data.drop('target', axis=1)
y_train = train_data['target']
X_test = test_data.copy()

# Обработка категориального признака C
# Преобразуем '+' и '-' в числа
le = LabelEncoder()
X_train['C'] = le.fit_transform(X_train['C'])
X_test['C'] = le.transform(X_test['C'])

# Обработка пропущенных значений
# Заполним пропуски медианой
for col in X_train.columns:
    if X_train[col].isnull().sum() > 0:
        median_val = X_train[col].median()
        X_train[col].fillna(median_val, inplace=True)
        X_test[col].fillna(median_val, inplace=True)

print(f"\nПосле обработки пропусков:")
print(f"Train NaN: {X_train.isnull().sum().sum()}")
print(f"Test NaN: {X_test.isnull().sum().sum()}")

# Модель 1: Random Forest
rf_model = RandomForestClassifier(
    n_estimators=300,
    max_depth=15,
    min_samples_split=5,
    min_samples_leaf=2,
    random_state=42,
    n_jobs=-1
)
rf_model.fit(X_train, y_train)
rf_pred_proba = rf_model.predict_proba(X_test)[:, 1]

# Модель 2: Gradient Boosting
gb_model = GradientBoostingClassifier(
    n_estimators=300,
    max_depth=5,
    learning_rate=0.05,
    random_state=42
)
gb_model.fit(X_train, y_train)
gb_pred_proba = gb_model.predict_proba(X_test)[:, 1]

# Ансамбль моделей
ensemble_pred_proba = (rf_pred_proba * 0.4 + gb_pred_proba * 0.6)

# Оценка моделей с помощью кросс-валидации
print("\nОценка моделей (ROC-AUC):")
rf_score = cross_val_score(rf_model, X_train, y_train, cv=5, scoring='roc_auc').mean()
print(f"Random Forest: {rf_score:.4f}")

gb_score = cross_val_score(gb_model, X_train, y_train, cv=5, scoring='roc_auc').mean()
print(f"Gradient Boosting: {gb_score:.4f}")

# Преобразуем вероятности в метки (0 или 1) для сохранения
# Используем порог 0.5
final_predictions = (ensemble_pred_proba >= 0.5).astype(int)

# Формирование результата в требуемом формате
result_df = pd.DataFrame({'target': final_predictions})

# Сохранение в CSV
result_df.to_csv('answers.csv', index=False)

print(f"\nПредсказания сохранены в файл 'answers.csv'")
print(f"Распределение предсказанных классов:")
print(result_df['target'].value_counts())
print(f"\nПример первых 10 предсказаний:")
print(result_df.head(10))

# Также сохраним вероятности для анализа
result_with_proba = pd.DataFrame({
    'target': final_predictions,
    'probability': ensemble_pred_proba
})
result_with_proba.to_csv('answers_with_proba.csv', index=False)
print(f"\nВероятности также сохранены в 'answers_with_proba.csv' для анализа")
