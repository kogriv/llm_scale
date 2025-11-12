import pandas as pd
import numpy as np
from sklearn.model_selection import StratifiedKFold, cross_val_score
from sklearn.preprocessing import LabelEncoder
from lightgbm import LGBMClassifier
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

# Квадраты важных признаков
for col in ['A', 'D', 'G', 'H', 'I']:
    X_train_fe[f'{col}^2'] = X_train[col] ** 2
    X_test_fe[f'{col}^2'] = X_test[col] ** 2

print(f"\nРазмер набора с новыми признаками: {X_train_fe.shape}")

# Кросс-валидация
cv = StratifiedKFold(n_splits=10, shuffle=True, random_state=42)

print("\n" + "="*70)
print("Тестирование различных конфигураций LightGBM")
print("="*70)

best_score = 0
best_model = None
best_config = None

# Тестируем различные конфигурации
configs = [
    # Больше деревьев, низкий learning rate
    {'n_estimators': 2000, 'learning_rate': 0.01, 'max_depth': 6, 'num_leaves': 40},
    {'n_estimators': 2000, 'learning_rate': 0.01, 'max_depth': 7, 'num_leaves': 50},
    {'n_estimators': 1500, 'learning_rate': 0.02, 'max_depth': 7, 'num_leaves': 50},
    {'n_estimators': 1500, 'learning_rate': 0.02, 'max_depth': 8, 'num_leaves': 60},
    
    # Средние параметры
    {'n_estimators': 1000, 'learning_rate': 0.03, 'max_depth': 7, 'num_leaves': 50},
    {'n_estimators': 1000, 'learning_rate': 0.05, 'max_depth': 6, 'num_leaves': 40},
    
    # Более глубокие деревья
    {'n_estimators': 1200, 'learning_rate': 0.02, 'max_depth': 10, 'num_leaves': 100},
    {'n_estimators': 1000, 'learning_rate': 0.03, 'max_depth': 12, 'num_leaves': 120},
]

for i, config in enumerate(configs, 1):
    print(f"\nКонфигурация {i}/{len(configs)}:")
    print(f"  n_estimators={config['n_estimators']}, lr={config['learning_rate']}, "
          f"depth={config['max_depth']}, leaves={config['num_leaves']}")
    
    lgbm = LGBMClassifier(
        **config,
        min_child_samples=10,
        subsample=0.8,
        colsample_bytree=0.8,
        reg_alpha=0.1,
        reg_lambda=0.1,
        random_state=42,
        verbose=-1,
        class_weight='balanced',
        n_jobs=-1
    )
    
    lgbm.fit(X_train_fe, y_train)
    score = cross_val_score(lgbm, X_train_fe, y_train, cv=cv, scoring='roc_auc', n_jobs=-1).mean()
    print(f"  ROC-AUC: {score:.4f}")
    
    if score > best_score:
        best_score = score
        best_model = lgbm
        best_config = config
        print(f"  ⭐ Новый лучший результат!")

print("\n" + "="*70)
print(f"ЛУЧШАЯ КОНФИГУРАЦИЯ:")
print(f"  n_estimators={best_config['n_estimators']}, lr={best_config['learning_rate']}, "
      f"depth={best_config['max_depth']}, leaves={best_config['num_leaves']}")
print(f"  ROC-AUC: {best_score:.4f}")
print("="*70)

# Предсказание с лучшей моделью
best_pred = best_model.predict_proba(X_test_fe)[:, 1]

# Преобразуем в метки
final_predictions = (best_pred >= 0.5).astype(int)

# Сохранение
result_df = pd.DataFrame({'target': final_predictions})
result_df.to_csv('answers.csv', index=False)

print(f"\n✓ Результаты сохранены в 'answers.csv'")
print(f"\nРаспределение предсказаний:")
print(f"  Класс 0: {(final_predictions==0).sum()} ({(final_predictions==0).sum()/len(final_predictions)*100:.1f}%)")
print(f"  Класс 1: {(final_predictions==1).sum()} ({(final_predictions==1).sum()/len(final_predictions)*100:.1f}%)")

result_with_proba = pd.DataFrame({
    'target': final_predictions,
    'probability': best_pred
})
result_with_proba.to_csv('answers_with_proba.csv', index=False)

print(f"\nСтатистика вероятностей:")
print(f"  Мин: {best_pred.min():.4f}")
print(f"  Макс: {best_pred.max():.4f}")
print(f"  Средн: {best_pred.mean():.4f}")
print(f"  Медиана: {np.median(best_pred):.4f}")

# Важность признаков
print(f"\nТоп-10 важнейших признаков:")
feature_importance = pd.DataFrame({
    'feature': X_train_fe.columns,
    'importance': best_model.feature_importances_
}).sort_values('importance', ascending=False)

for idx, row in feature_importance.head(10).iterrows():
    print(f"  {row['feature']:15s}: {row['importance']:.0f}")
