import pandas as pd
import numpy as np
from sklearn.ensemble import (RandomForestClassifier, GradientBoostingClassifier, 
                             ExtraTreesClassifier, AdaBoostClassifier, HistGradientBoostingClassifier)
from sklearn.model_selection import cross_val_score, StratifiedKFold
from sklearn.preprocessing import LabelEncoder
from sklearn.tree import DecisionTreeClassifier
import warnings
warnings.filterwarnings('ignore')

# Загрузка данных
train_data = pd.read_csv('data/task2/train.csv')
test_data = pd.read_csv('data/task2/test.csv')

print(f"Размер обучающей выборки: {train_data.shape}")
print(f"Размер тестовой выборки: {test_data.shape}")

# Разделение на признаки и целевую переменную
X_train = train_data.drop('target', axis=1).copy()
y_train = train_data['target'].copy()
X_test = test_data.copy()

print(f"\nБаланс классов: 0={(y_train==0).sum()}, 1={(y_train==1).sum()}")

# Обработка категориального признака C
le = LabelEncoder()
X_train['C'] = le.fit_transform(X_train['C'])
X_test['C'] = le.transform(X_test['C'])

# Обработка пропущенных значений
for col in X_train.columns:
    if X_train[col].isnull().sum() > 0:
        median_val = X_train[col].median()
        X_train.loc[:, col] = X_train[col].fillna(median_val)
        X_test.loc[:, col] = X_test[col].fillna(median_val)

# Feature Engineering - создаем много новых признаков
X_train_fe = X_train.copy()
X_test_fe = X_test.copy()

# Попарные произведения всех признаков
for i, col1 in enumerate(X_train.columns):
    for col2 in list(X_train.columns)[i+1:]:
        X_train_fe[f'{col1}*{col2}'] = X_train[col1] * X_train[col2]
        X_test_fe[f'{col1}*{col2}'] = X_test[col1] * X_test[col2]

# Квадраты всех признаков
for col in X_train.columns:
    X_train_fe[f'{col}^2'] = X_train[col] ** 2
    X_test_fe[f'{col}^2'] = X_test[col] ** 2

# Кубы важных признаков
for col in ['A', 'D', 'G', 'H', 'I']:
    X_train_fe[f'{col}^3'] = X_train[col] ** 3
    X_test_fe[f'{col}^3'] = X_test[col] ** 3

# Логарифмы (избегаем log(0))
for col in X_train.columns:
    X_train_fe[f'log_{col}'] = np.log1p(np.abs(X_train[col]))
    X_test_fe[f'log_{col}'] = np.log1p(np.abs(X_test[col]))

# Экспоненты небольших значений
for col in ['A', 'B', 'C', 'E', 'F']:
    X_train_fe[f'exp_{col}'] = np.exp(X_train[col] / 10)
    X_test_fe[f'exp_{col}'] = np.exp(X_test[col] / 10)

print(f"\nРазмер набора с новыми признаками: {X_train_fe.shape}")

# Кросс-валидация
cv = StratifiedKFold(n_splits=10, shuffle=True, random_state=42)

print("\n" + "="*60)
print("Обучение и оценка моделей...")
print("="*60)

models = []
scores_list = []
predictions = []

# 1. AdaBoost с разными base_estimator
print("\n1. AdaBoost модели:")
for depth in [1, 2, 3]:
    base = DecisionTreeClassifier(max_depth=depth, random_state=42)
    ada = AdaBoostClassifier(
        estimator=base,
        n_estimators=500,
        learning_rate=0.8,
        random_state=42
    )
    ada.fit(X_train_fe, y_train)
    score = cross_val_score(ada, X_train_fe, y_train, cv=cv, scoring='roc_auc', n_jobs=-1).mean()
    print(f"   AdaBoost (depth={depth}): {score:.4f}")
    
    if score > 0.85:  # Используем только хорошие модели
        models.append(f"AdaBoost_d{depth}")
        scores_list.append(score)
        predictions.append(ada.predict_proba(X_test_fe)[:, 1])

# 2. Gradient Boosting варианты
print("\n2. Gradient Boosting модели:")
for lr, depth in [(0.01, 5), (0.02, 6), (0.05, 4)]:
    gb = GradientBoostingClassifier(
        n_estimators=1000,
        max_depth=depth,
        learning_rate=lr,
        subsample=0.8,
        min_samples_split=5,
        random_state=42
    )
    gb.fit(X_train_fe, y_train)
    score = cross_val_score(gb, X_train_fe, y_train, cv=cv, scoring='roc_auc', n_jobs=-1).mean()
    print(f"   GB (lr={lr}, depth={depth}): {score:.4f}")
    
    if score > 0.85:
        models.append(f"GB_lr{lr}_d{depth}")
        scores_list.append(score)
        predictions.append(gb.predict_proba(X_test_fe)[:, 1])

# 3. HistGradientBoosting - быстрый и мощный
print("\n3. HistGradientBoosting модели:")
for lr, depth in [(0.05, 10), (0.1, 8), (0.02, 15)]:
    hgb = HistGradientBoostingClassifier(
        max_iter=500,
        max_depth=depth,
        learning_rate=lr,
        random_state=42
    )
    hgb.fit(X_train_fe, y_train)
    score = cross_val_score(hgb, X_train_fe, y_train, cv=cv, scoring='roc_auc', n_jobs=-1).mean()
    print(f"   HistGB (lr={lr}, depth={depth}): {score:.4f}")
    
    if score > 0.85:
        models.append(f"HistGB_lr{lr}_d{depth}")
        scores_list.append(score)
        predictions.append(hgb.predict_proba(X_test_fe)[:, 1])

# 4. Random Forest и Extra Trees
print("\n4. Ансамбли деревьев:")
for n_est, max_d in [(1000, 30), (800, 25), (600, 20)]:
    rf = RandomForestClassifier(
        n_estimators=n_est,
        max_depth=max_d,
        min_samples_split=3,
        min_samples_leaf=1,
        max_features='sqrt',
        random_state=42,
        n_jobs=-1,
        class_weight='balanced_subsample'
    )
    rf.fit(X_train_fe, y_train)
    score = cross_val_score(rf, X_train_fe, y_train, cv=cv, scoring='roc_auc', n_jobs=-1).mean()
    print(f"   RF (n={n_est}, d={max_d}): {score:.4f}")
    
    if score > 0.85:
        models.append(f"RF_n{n_est}_d{max_d}")
        scores_list.append(score)
        predictions.append(rf.predict_proba(X_test_fe)[:, 1])

print("\n" + "="*60)
print(f"Отобрано {len(models)} моделей для ансамбля")
print("="*60)

# Взвешенный ансамбль - экспоненциальные веса
scores_array = np.array(scores_list)
exp_scores = np.exp((scores_array - scores_array.min()) * 20)
weights = exp_scores / exp_scores.sum()

print("\nТоп-5 моделей по весу:")
top_idx = np.argsort(weights)[::-1][:5]
for i in top_idx:
    print(f"  {models[i]}: score={scores_list[i]:.4f}, weight={weights[i]:.4f}")

# Финальное предсказание
predictions_array = np.array(predictions)
ensemble_pred = np.average(predictions_array, axis=0, weights=weights)

# Преобразуем в метки
final_predictions = (ensemble_pred >= 0.5).astype(int)

# Сохранение
result_df = pd.DataFrame({'target': final_predictions})
result_df.to_csv('answers.csv', index=False)

print(f"\n✓ Результаты сохранены в 'answers.csv'")
print(f"\nРаспределение предсказаний:")
print(f"  Класс 0: {(final_predictions==0).sum()}")
print(f"  Класс 1: {(final_predictions==1).sum()}")

result_with_proba = pd.DataFrame({
    'target': final_predictions,
    'probability': ensemble_pred
})
result_with_proba.to_csv('answers_with_proba.csv', index=False)

print(f"\nСтатистика вероятностей:")
print(f"  Мин: {ensemble_pred.min():.4f}")
print(f"  Макс: {ensemble_pred.max():.4f}")
print(f"  Средн: {ensemble_pred.mean():.4f}")

print("\n" + "="*60)
print(f"Максимальный AUC на CV: {max(scores_list):.4f}")
print(f"Средний AUC моделей в ансамбле: {np.mean(scores_list):.4f}")
print("="*60)
