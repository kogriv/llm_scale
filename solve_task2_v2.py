import pandas as pd
import numpy as np
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier, ExtraTreesClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import cross_val_score, StratifiedKFold
from sklearn.preprocessing import LabelEncoder, StandardScaler
from sklearn.metrics import roc_auc_score

# Загрузка данных
train_data = pd.read_csv('data/task2/train.csv')
test_data = pd.read_csv('data/task2/test.csv')

print(f"Размер обучающей выборки: {train_data.shape}")
print(f"Размер тестовой выборки: {test_data.shape}")

# Разделение на признаки и целевую переменную
X_train = train_data.drop('target', axis=1).copy()
y_train = train_data['target'].copy()
X_test = test_data.copy()

# Обработка категориального признака C
le = LabelEncoder()
X_train['C'] = le.fit_transform(X_train['C'])
X_test['C'] = le.transform(X_test['C'])

# Обработка пропущенных значений - заполним медианой
for col in X_train.columns:
    if X_train[col].isnull().sum() > 0:
        median_val = X_train[col].median()
        X_train.loc[:, col] = X_train[col].fillna(median_val)
        X_test.loc[:, col] = X_test[col].fillna(median_val)

print(f"\nРаспределение классов:")
print(y_train.value_counts(normalize=True))

# Создадим дополнительные признаки
X_train_enhanced = X_train.copy()
X_test_enhanced = X_test.copy()

# Добавим взаимодействия между признаками
X_train_enhanced['A_D'] = X_train['A'] * X_train['D']
X_test_enhanced['A_D'] = X_test['A'] * X_test['D']

X_train_enhanced['G_H'] = X_train['G'] * X_train['H']
X_test_enhanced['G_H'] = X_test['G'] * X_test['H']

X_train_enhanced['E_F'] = X_train['E'] * X_train['F']
X_test_enhanced['E_F'] = X_test['E'] * X_test['F']

# Модель 1: Random Forest с оптимизированными параметрами
rf_model = RandomForestClassifier(
    n_estimators=500,
    max_depth=20,
    min_samples_split=10,
    min_samples_leaf=4,
    max_features='sqrt',
    random_state=42,
    n_jobs=-1,
    class_weight='balanced'
)

# Модель 2: Gradient Boosting
gb_model = GradientBoostingClassifier(
    n_estimators=500,
    max_depth=4,
    learning_rate=0.03,
    subsample=0.8,
    random_state=42
)

# Модель 3: Extra Trees
et_model = ExtraTreesClassifier(
    n_estimators=500,
    max_depth=20,
    min_samples_split=10,
    min_samples_leaf=4,
    random_state=42,
    n_jobs=-1,
    class_weight='balanced'
)

# Модель 4: Logistic Regression (на стандартизированных данных)
scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train_enhanced)
X_test_scaled = scaler.transform(X_test_enhanced)

lr_model = LogisticRegression(
    max_iter=1000,
    C=0.1,
    random_state=42,
    class_weight='balanced'
)

# Обучение моделей
print("\nОбучение моделей...")
rf_model.fit(X_train_enhanced, y_train)
gb_model.fit(X_train_enhanced, y_train)
et_model.fit(X_train_enhanced, y_train)
lr_model.fit(X_train_scaled, y_train)

# Предсказания
rf_pred = rf_model.predict_proba(X_test_enhanced)[:, 1]
gb_pred = gb_model.predict_proba(X_test_enhanced)[:, 1]
et_pred = et_model.predict_proba(X_test_enhanced)[:, 1]
lr_pred = lr_model.predict_proba(X_test_scaled)[:, 1]

# Оценка моделей
cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)

print("\nОценка моделей (ROC-AUC на кросс-валидации):")
rf_score = cross_val_score(rf_model, X_train_enhanced, y_train, cv=cv, scoring='roc_auc').mean()
print(f"Random Forest: {rf_score:.4f}")

gb_score = cross_val_score(gb_model, X_train_enhanced, y_train, cv=cv, scoring='roc_auc').mean()
print(f"Gradient Boosting: {gb_score:.4f}")

et_score = cross_val_score(et_model, X_train_enhanced, y_train, cv=cv, scoring='roc_auc').mean()
print(f"Extra Trees: {et_score:.4f}")

lr_score = cross_val_score(lr_model, X_train_scaled, y_train, cv=cv, scoring='roc_auc').mean()
print(f"Logistic Regression: {lr_score:.4f}")

# Взвешенный ансамбль на основе производительности моделей
# Больший вес лучшим моделям
weights = np.array([rf_score, gb_score, et_score, lr_score])
weights = weights / weights.sum()

ensemble_pred = (rf_pred * weights[0] + 
                gb_pred * weights[1] + 
                et_pred * weights[2] + 
                lr_pred * weights[3])

print(f"\nВеса моделей в ансамбле:")
print(f"Random Forest: {weights[0]:.4f}")
print(f"Gradient Boosting: {weights[1]:.4f}")
print(f"Extra Trees: {weights[2]:.4f}")
print(f"Logistic Regression: {weights[3]:.4f}")

# Преобразуем вероятности в метки
final_predictions = (ensemble_pred >= 0.5).astype(int)

# Сохранение результата
result_df = pd.DataFrame({'target': final_predictions})
result_df.to_csv('answers.csv', index=False)

print(f"\nПредсказания сохранены в файл 'answers.csv'")
print(f"Распределение предсказанных классов:")
print(result_df['target'].value_counts())

# Также сохраним вероятности
result_with_proba = pd.DataFrame({
    'target': final_predictions,
    'probability': ensemble_pred
})
result_with_proba.to_csv('answers_with_proba.csv', index=False)
print(f"\nСтатистика вероятностей:")
print(f"Мин: {ensemble_pred.min():.4f}")
print(f"Макс: {ensemble_pred.max():.4f}")
print(f"Средн: {ensemble_pred.mean():.4f}")
