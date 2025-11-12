import pandas as pd
import numpy as np
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier, ExtraTreesClassifier, AdaBoostClassifier
from sklearn.model_selection import cross_val_score, StratifiedKFold
from sklearn.preprocessing import LabelEncoder, StandardScaler, PolynomialFeatures
from sklearn.linear_model import LogisticRegression
from sklearn.svm import SVC
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

print(f"\nБаланс классов:")
print(y_train.value_counts())
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

# Создадим расширенный набор признаков
X_train_enhanced = X_train.copy()
X_test_enhanced = X_test.copy()

# Добавим попарные взаимодействия важных признаков
feature_pairs = [
    ('A', 'D'), ('A', 'B'), ('A', 'I'),
    ('B', 'E'), ('B', 'F'),
    ('D', 'G'), ('D', 'H'),
    ('E', 'F'), ('E', 'I'),
    ('G', 'H'), ('G', 'I'),
    ('H', 'I')
]

for f1, f2 in feature_pairs:
    X_train_enhanced[f'{f1}_{f2}'] = X_train[f1] * X_train[f2]
    X_test_enhanced[f'{f1}_{f2}'] = X_test[f1] * X_test[f2]

# Добавим квадраты признаков
for col in ['A', 'D', 'G', 'H', 'I']:
    X_train_enhanced[f'{col}_sq'] = X_train[col] ** 2
    X_test_enhanced[f'{col}_sq'] = X_test[col] ** 2

# Добавим отношения
X_train_enhanced['A_D_ratio'] = X_train['A'] / (X_train['D'] + 0.001)
X_test_enhanced['A_D_ratio'] = X_test['A'] / (X_test['D'] + 0.001)

X_train_enhanced['G_I_ratio'] = X_train['G'] / (X_train['I'] + 0.001)
X_test_enhanced['G_I_ratio'] = X_test['G'] / (X_test['I'] + 0.001)

print(f"\nРазмер расширенного набора признаков: {X_train_enhanced.shape}")

# Подготовка для моделей
cv = StratifiedKFold(n_splits=10, shuffle=True, random_state=42)

# Модель 1: Random Forest с увеличенными параметрами
print("\nОбучение моделей...")
rf_model = RandomForestClassifier(
    n_estimators=800,
    max_depth=25,
    min_samples_split=5,
    min_samples_leaf=2,
    max_features='sqrt',
    random_state=42,
    n_jobs=-1,
    class_weight='balanced_subsample'
)
rf_model.fit(X_train_enhanced, y_train)
rf_pred = rf_model.predict_proba(X_test_enhanced)[:, 1]

# Модель 2: Gradient Boosting - более глубокая
gb_model = GradientBoostingClassifier(
    n_estimators=800,
    max_depth=6,
    learning_rate=0.02,
    subsample=0.8,
    min_samples_split=5,
    min_samples_leaf=2,
    random_state=42
)
gb_model.fit(X_train_enhanced, y_train)
gb_pred = gb_model.predict_proba(X_test_enhanced)[:, 1]

# Модель 3: Extra Trees
et_model = ExtraTreesClassifier(
    n_estimators=800,
    max_depth=25,
    min_samples_split=5,
    min_samples_leaf=2,
    random_state=42,
    n_jobs=-1,
    class_weight='balanced_subsample'
)
et_model.fit(X_train_enhanced, y_train)
et_pred = et_model.predict_proba(X_test_enhanced)[:, 1]

# Модель 4: AdaBoost
ada_model = AdaBoostClassifier(
    n_estimators=300,
    learning_rate=0.5,
    random_state=42
)
ada_model.fit(X_train_enhanced, y_train)
ada_pred = ada_model.predict_proba(X_test_enhanced)[:, 1]

# Модель 5: Logistic Regression на стандартизированных данных
scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train_enhanced)
X_test_scaled = scaler.transform(X_test_enhanced)

lr_model = LogisticRegression(
    max_iter=2000,
    C=0.05,
    penalty='l2',
    random_state=42,
    class_weight='balanced'
)
lr_model.fit(X_train_scaled, y_train)
lr_pred = lr_model.predict_proba(X_test_scaled)[:, 1]

# Оценка моделей
print("\nОценка моделей (ROC-AUC на 10-fold CV):")
scores = []

rf_score = cross_val_score(rf_model, X_train_enhanced, y_train, cv=cv, scoring='roc_auc', n_jobs=-1).mean()
print(f"Random Forest: {rf_score:.4f}")
scores.append(rf_score)

gb_score = cross_val_score(gb_model, X_train_enhanced, y_train, cv=cv, scoring='roc_auc', n_jobs=-1).mean()
print(f"Gradient Boosting: {gb_score:.4f}")
scores.append(gb_score)

et_score = cross_val_score(et_model, X_train_enhanced, y_train, cv=cv, scoring='roc_auc', n_jobs=-1).mean()
print(f"Extra Trees: {et_score:.4f}")
scores.append(et_score)

ada_score = cross_val_score(ada_model, X_train_enhanced, y_train, cv=cv, scoring='roc_auc', n_jobs=-1).mean()
print(f"AdaBoost: {ada_score:.4f}")
scores.append(ada_score)

lr_score = cross_val_score(lr_model, X_train_scaled, y_train, cv=cv, scoring='roc_auc', n_jobs=-1).mean()
print(f"Logistic Regression: {lr_score:.4f}")
scores.append(lr_score)

# Взвешенный ансамбль - больший вес лучшим моделям
scores = np.array(scores)
# Применим softmax к оценкам для получения весов
exp_scores = np.exp(scores * 10)  # умножаем на 10 для большей разницы
weights = exp_scores / exp_scores.sum()

print(f"\nВеса моделей в ансамбле:")
print(f"Random Forest: {weights[0]:.4f}")
print(f"Gradient Boosting: {weights[1]:.4f}")
print(f"Extra Trees: {weights[2]:.4f}")
print(f"AdaBoost: {weights[3]:.4f}")
print(f"Logistic Regression: {weights[4]:.4f}")

# Финальный ансамбль
ensemble_pred = (rf_pred * weights[0] + 
                gb_pred * weights[1] + 
                et_pred * weights[2] + 
                ada_pred * weights[3] + 
                lr_pred * weights[4])

# Преобразуем вероятности в метки
final_predictions = (ensemble_pred >= 0.5).astype(int)

# Сохранение результата
result_df = pd.DataFrame({'target': final_predictions})
result_df.to_csv('answers.csv', index=False)

print(f"\n✓ Предсказания сохранены в файл 'answers.csv'")
print(f"Распределение предсказанных классов:")
print(result_df['target'].value_counts())
print(f"Пропорция класса 1: {final_predictions.mean():.3f}")

# Также сохраним вероятности
result_with_proba = pd.DataFrame({
    'target': final_predictions,
    'probability': ensemble_pred
})
result_with_proba.to_csv('answers_with_proba.csv', index=False)

print(f"\nСтатистика вероятностей ансамбля:")
print(f"Мин: {ensemble_pred.min():.4f}")
print(f"Макс: {ensemble_pred.max():.4f}")
print(f"Средн: {ensemble_pred.mean():.4f}")
print(f"Медиана: {np.median(ensemble_pred):.4f}")

print(f"\n{'='*60}")
print(f"Ожидаемый AUC на тесте: ~{max(scores):.4f} (на основе лучшей модели)")
print(f"{'='*60}")
