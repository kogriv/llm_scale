import pandas as pd
import numpy as np
from sklearn.model_selection import StratifiedKFold, cross_val_score
from sklearn.preprocessing import LabelEncoder
from lightgbm import LGBMClassifier
from catboost import CatBoostClassifier
from xgboost import XGBClassifier
from sklearn.ensemble import GradientBoostingClassifier, AdaBoostClassifier
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
print(f"Пропорция класса 1: {y_train.mean():.3f}")

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

# Feature Engineering
X_train_fe = X_train.copy()
X_test_fe = X_test.copy()

# Взаимодействия признаков
important_pairs = [
    ('A', 'D'), ('A', 'I'), ('A', 'G'), ('A', 'H'),
    ('B', 'E'), ('B', 'F'), ('D', 'G'), ('D', 'H'),
    ('E', 'F'), ('E', 'I'), ('G', 'H'), ('G', 'I'), ('H', 'I')
]

for f1, f2 in important_pairs:
    X_train_fe[f'{f1}*{f2}'] = X_train[f1] * X_train[f2]
    X_test_fe[f'{f1}*{f2}'] = X_test[f1] * X_test[f2]
    X_train_fe[f'{f1}/{f2}'] = X_train[f1] / (X_train[f2] + 0.001)
    X_test_fe[f'{f1}/{f2}'] = X_test[f1] / (X_test[f2] + 0.001)

# Квадраты и кубы
for col in ['A', 'D', 'G', 'H', 'I']:
    X_train_fe[f'{col}^2'] = X_train[col] ** 2
    X_test_fe[f'{col}^2'] = X_test[col] ** 2
    X_train_fe[f'{col}^3'] = X_train[col] ** 3
    X_test_fe[f'{col}^3'] = X_test[col] ** 3

print(f"\nРазмер набора с новыми признаками: {X_train_fe.shape}")

# Кросс-валидация
cv = StratifiedKFold(n_splits=10, shuffle=True, random_state=42)

print("\n" + "="*70)
print("Обучение и оценка моделей")
print("="*70)

models_list = []
scores_list = []
predictions_list = []

# 1. LightGBM - несколько конфигураций
print("\n1. LightGBM модели:")
lgbm_configs = [
    {'n_estimators': 1000, 'learning_rate': 0.03, 'max_depth': 7, 'num_leaves': 50},
    {'n_estimators': 1500, 'learning_rate': 0.02, 'max_depth': 8, 'num_leaves': 60},
    {'n_estimators': 800, 'learning_rate': 0.05, 'max_depth': 6, 'num_leaves': 40},
]

for i, config in enumerate(lgbm_configs):
    lgbm = LGBMClassifier(
        **config,
        min_child_samples=10,
        subsample=0.8,
        colsample_bytree=0.8,
        random_state=42,
        verbose=-1,
        class_weight='balanced'
    )
    lgbm.fit(X_train_fe, y_train)
    score = cross_val_score(lgbm, X_train_fe, y_train, cv=cv, scoring='roc_auc', n_jobs=-1).mean()
    print(f"   LightGBM #{i+1} (n={config['n_estimators']}, lr={config['learning_rate']}, d={config['max_depth']}): {score:.4f}")
    
    models_list.append(f"LGBM_{i+1}")
    scores_list.append(score)
    predictions_list.append(lgbm.predict_proba(X_test_fe)[:, 1])

# 2. CatBoost - лучший для категориальных признаков
print("\n2. CatBoost модели:")
catboost_configs = [
    {'iterations': 1000, 'learning_rate': 0.05, 'depth': 6},
    {'iterations': 1500, 'learning_rate': 0.03, 'depth': 7},
    {'iterations': 800, 'learning_rate': 0.08, 'depth': 5},
]

for i, config in enumerate(catboost_configs):
    catb = CatBoostClassifier(
        **config,
        l2_leaf_reg=3,
        random_state=42,
        verbose=0,
        auto_class_weights='Balanced'
    )
    catb.fit(X_train_fe, y_train)
    score = cross_val_score(catb, X_train_fe, y_train, cv=cv, scoring='roc_auc', n_jobs=-1).mean()
    print(f"   CatBoost #{i+1} (n={config['iterations']}, lr={config['learning_rate']}, d={config['depth']}): {score:.4f}")
    
    models_list.append(f"CatBoost_{i+1}")
    scores_list.append(score)
    predictions_list.append(catb.predict_proba(X_test_fe)[:, 1])

# 3. XGBoost
print("\n3. XGBoost модели:")
xgb_configs = [
    {'n_estimators': 1000, 'learning_rate': 0.03, 'max_depth': 6},
    {'n_estimators': 1200, 'learning_rate': 0.025, 'max_depth': 7},
    {'n_estimators': 800, 'learning_rate': 0.05, 'max_depth': 5},
]

scale_pos_weight = (y_train == 0).sum() / (y_train == 1).sum()

for i, config in enumerate(xgb_configs):
    xgb = XGBClassifier(
        **config,
        min_child_weight=3,
        subsample=0.8,
        colsample_bytree=0.8,
        random_state=42,
        scale_pos_weight=scale_pos_weight,
        eval_metric='logloss'
    )
    xgb.fit(X_train_fe, y_train)
    score = cross_val_score(xgb, X_train_fe, y_train, cv=cv, scoring='roc_auc', n_jobs=-1).mean()
    print(f"   XGBoost #{i+1} (n={config['n_estimators']}, lr={config['learning_rate']}, d={config['max_depth']}): {score:.4f}")
    
    models_list.append(f"XGB_{i+1}")
    scores_list.append(score)
    predictions_list.append(xgb.predict_proba(X_test_fe)[:, 1])

# 4. Sklearn Gradient Boosting для разнообразия
print("\n4. Sklearn GradientBoosting:")
gb = GradientBoostingClassifier(
    n_estimators=1000,
    max_depth=6,
    learning_rate=0.03,
    subsample=0.8,
    random_state=42
)
gb.fit(X_train_fe, y_train)
gb_score = cross_val_score(gb, X_train_fe, y_train, cv=cv, scoring='roc_auc', n_jobs=-1).mean()
print(f"   GradientBoosting: {gb_score:.4f}")

models_list.append("GB")
scores_list.append(gb_score)
predictions_list.append(gb.predict_proba(X_test_fe)[:, 1])

print("\n" + "="*70)
print(f"Всего моделей: {len(models_list)}")
print("="*70)

# Взвешенный ансамбль - экспоненциальные веса для лучших моделей
scores_array = np.array(scores_list)
# Используем softmax с температурой для весов
temperature = 15
exp_scores = np.exp((scores_array - scores_array.min()) * temperature)
weights = exp_scores / exp_scores.sum()

print("\nТоп-5 моделей по весу в ансамбле:")
top_idx = np.argsort(weights)[::-1][:5]
for idx in top_idx:
    print(f"  {models_list[idx]:15s}: AUC={scores_list[idx]:.4f}, вес={weights[idx]:.4f}")

# Финальное предсказание
predictions_array = np.array(predictions_list)
ensemble_pred = np.average(predictions_array, axis=0, weights=weights)

# Преобразуем в метки
final_predictions = (ensemble_pred >= 0.5).astype(int)

# Сохранение
result_df = pd.DataFrame({'target': final_predictions})
result_df.to_csv('answers.csv', index=False)

print(f"\n✓ Результаты сохранены в 'answers.csv'")
print(f"\nРаспределение предсказаний:")
print(f"  Класс 0: {(final_predictions==0).sum()} ({(final_predictions==0).sum()/len(final_predictions)*100:.1f}%)")
print(f"  Класс 1: {(final_predictions==1).sum()} ({(final_predictions==1).sum()/len(final_predictions)*100:.1f}%)")

result_with_proba = pd.DataFrame({
    'target': final_predictions,
    'probability': ensemble_pred
})
result_with_proba.to_csv('answers_with_proba.csv', index=False)

print(f"\nСтатистика вероятностей ансамбля:")
print(f"  Мин: {ensemble_pred.min():.4f}")
print(f"  Макс: {ensemble_pred.max():.4f}")
print(f"  Средн: {ensemble_pred.mean():.4f}")
print(f"  Медиана: {np.median(ensemble_pred):.4f}")

print("\n" + "="*70)
print(f"МАКСИМАЛЬНЫЙ AUC НА CV: {max(scores_list):.4f}")
print(f"СРЕДНИЙ AUC ВСЕХ МОДЕЛЕЙ: {np.mean(scores_list):.4f}")
print(f"СРЕДНИЙ AUC ТОП-5 МОДЕЛЕЙ: {np.mean([scores_list[i] for i in top_idx]):.4f}")
print("="*70)
