import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.neighbors import KNeighborsClassifier
from sklearn.metrics import confusion_matrix, classification_report, accuracy_score
import warnings
warnings.filterwarnings('ignore')

# Определяем структуру файла данных согласно описанию
column_names = [
    'Vehicle_Name',
    'Sports_Car',
    'SUV',
    'Wagon',
    'Minivan',
    'Pickup',
    'AWD',
    'RWD',
    'Retail_Price',
    'Dealer_Cost',
    'Engine_Size',
    'Cylinders',
    'Horsepower',
    'City_MPG',
    'Highway_MPG',
    'Weight',
    'Wheel_Base',
    'Length',
    'Width',
]

# Загружаем данные
# Используем разделитель ';' так как данные разделены точкой с запятой
df = pd.read_csv('04cars.dat', sep=';', names=column_names, encoding='latin-1')

print("Первые 5 записей датасета:")
print(df.head())

# Проверяем распределение целевой переменной
print("Распределение классов:")
print(f"Спортивные автомобили: {df['Sports_Car'].sum()} ({df['Sports_Car'].sum()/len(df)*100:.1f}%)")
print(f"Не спортивные автомобили: {(1-df['Sports_Car']).sum()} ({(1-df['Sports_Car']).sum()/len(df)*100:.1f}%)\n")

# Выделение признаков и разделение на выборки

# Выбираем признаки для классификации
feature_columns = [
    'AWD',                # Полный привод
    'RWD',                # Задний привод
    'Retail_Price',       # Розничная цена
    'Dealer_Cost',        # Дилерская стоимость
    'Engine_Size',        # Объем двигателя
    'Cylinders',          # Количество цилиндров
    'Horsepower',         # Мощность
    'City_MPG',           # Расход в городе
    'Highway_MPG',        # Расход на трассе
    'Weight',             # Вес
    'Wheel_Base',         # Колесная база
    'Length',             # Длина
    'Width'               # Ширина
]

# Целевая переменная
target_column = 'Sports_Car'

# Удаляем строки с пропущенными значениями в используемых признаках
df_clean = df[feature_columns + [target_column]].dropna()

# Разделяем на признаки (X) и целевую переменную (y)
X = df_clean[feature_columns]
y = df_clean[target_column]

print("Используемые признаки для классификации:")
for i, col in enumerate(feature_columns, 1):
    print(f"  {i}. {col}")
print()

# Разделяем данные на обучающую и тестовую выборки
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.35, stratify=y, random_state=52
)

print(f"Обучающая выборка: {X_train.shape[0]} наблюдений")
print(f"  - Спортивные: {y_train.sum()}")
print(f"  - Не спортивные: {(1-y_train).sum()}")
print()
print(f"Тестовая выборка: {X_test.shape[0]} наблюдений")
print(f"  - Спортивные: {y_test.sum()}")
print(f"  - Не спортивные: {(1-y_test).sum()}")
print()

# Нормализация данных
scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled = scaler.transform(X_test)

# Определение наилучшего значения k
k_range = range(1, 31)
train_scores = []
test_scores = []

for k in k_range:
    # Создаем и обучаем модель k-NN
    knn = KNeighborsClassifier(n_neighbors=k)
    knn.fit(X_train_scaled, y_train)

    # Вычисляем точность на обучающей и тестовой выборках
    train_score = knn.score(X_train_scaled, y_train)
    test_score = knn.score(X_test_scaled, y_test)

    train_scores.append(train_score)
    test_scores.append(test_score)

    print(f"k={k:2d}: Обучающая точность = {train_score:.4f}")
    print(f"Тестовая точность = {test_score:.4f}")

print()

# Находим оптимальное значение k
best_k = k_range[np.argmax(test_scores)]
best_test_score = max(test_scores)

print(f"Наилучшее значение k = {best_k}")
print(f"Точность на тестовой выборке: {best_test_score:.4f} ({best_test_score*100:.2f}%)")
print()

# Визуализация зависимости точности от k
plt.figure(figsize=(12, 6))
plt.plot(k_range, train_scores, 'b-', label='Обучающая выборка', linewidth=2)
plt.plot(k_range, test_scores, 'r-', label='Тестовая выборка', linewidth=2)
plt.plot(best_k, best_test_score, 'go', markersize=12,
         label=f'Оптимальное k={best_k}')
plt.xlabel('Значение k (количество соседей)', fontsize=12)
plt.ylabel('Точность классификации', fontsize=12)
plt.title('Зависимость точности классификации от параметра k', fontsize=14, fontweight='bold')
plt.legend(fontsize=11)
plt.grid(True, alpha=0.3)
plt.xticks(range(0, 31, 5))
plt.tight_layout()
plt.plot()

# Создаем финальную модель с наилучшим k
final_knn = KNeighborsClassifier(n_neighbors=best_k)
final_knn.fit(X_train_scaled, y_train)

# Делаем прогноз на тестовой выборке
y_pred = final_knn.predict(X_test_scaled)

# Вычисляем матрицу сопряженности (confusion matrix)
cm = confusion_matrix(y_test, y_pred)

print("Таблица сопряженности")
print("-" * 50)
print(f"{'':20s} | {'Предсказано: 0':>15s} | {'Предсказано: 1':>15s}")
print("-" * 50)
print(f"{'Реально: 0 (не спорт)':20s} | {cm[0,0]:15d} | {cm[0,1]:15d}")
print(f"{'Реально: 1 (спорт)':20s} | {cm[1,0]:15d} | {cm[1,1]:15d}")
print("-" * 50)
print()

# Расшифровка матрицы сопряженности
tn, fp, fn, tp = cm.ravel()
print("Расшифровка результатов:")
print(f"True Negatives:  {tn} - правильно классифицированы как НЕ спортивные")
print(f"False Positives: {fp} - ошибочно классифицированы как спортивные")
print(f"False Negatives: {fn} - ошибочно классифицированы как НЕ спортивные")
print(f"True Positives:  {tp} - правильно классифицированы как спортивные")
print()

# Вычисляем метрики качества
accuracy = accuracy_score(y_test, y_pred)
error_rate = 1 - accuracy

# Дополнительные метрики
precision = tp / (tp + fp) if (tp + fp) > 0 else 0
recall = tp / (tp + fn) if (tp + fn) > 0 else 0

print(f"Точность (Accuracy):     {accuracy:.4f} ({accuracy*100:.2f}%)")
print(f"ПРОЦЕНТ ОШИБОК:          {error_rate:.4f} ({error_rate*100:.2f}%)")
print(f"Precision (Точность):    {precision:.4f} - доля правильных среди предсказанных как спортивные")
print(f"Recall (Полнота):        {recall:.4f} - доля найденных спортивных автомобилей")

report = classification_report(
    y_test, y_pred, target_names=['Не спортивный', 'Спортивный'], digits=4
)
print(report)

plt.figure(figsize=(10, 8))
plt.imshow(cm, interpolation='nearest', cmap=plt.cm.Blues)
plt.title(f'Матрица сопряженности (k={best_k})', fontsize=16, fontweight='bold')
plt.colorbar()

classes = ['Не спортивный', 'Спортивный']
tick_marks = np.arange(len(classes))
plt.xticks(tick_marks, classes, fontsize=12)
plt.yticks(tick_marks, classes, fontsize=12)

thresh = cm.max() / 2.
for i in range(cm.shape[0]):
    for j in range(cm.shape[1]):
        plt.text(
            j, i, format(cm[i, j], 'd'),
            ha="center", va="center", fontsize=20,
            color="white" if cm[i, j] > thresh else "black"
        )

plt.ylabel('Реальный класс', fontsize=13)
plt.xlabel('Предсказанный класс', fontsize=13)
plt.tight_layout()
plt.show()

# Важность характеристик для классификации
# Сравниваем средние значения признаков для спортивных и не спортивных авто
sports_cars = df_clean[df_clean['Sports_Car'] == 1][feature_columns]
non_sports_cars = df_clean[df_clean['Sports_Car'] == 0][feature_columns]

print("\nСредние значения характеристик:\n")
print(f"{'Признак':<20s} | {'Спортивные':>15s} | {'Не спортивные':>15s} | {'Разница':>15s}")
print("-" * 70)

for col in feature_columns:
    sports_mean = sports_cars[col].mean()
    non_sports_mean = non_sports_cars[col].mean()
    diff = sports_mean - non_sports_mean
    print(f"{col:<20s} | {sports_mean:>15.2f} | {non_sports_mean:>15.2f} | {diff:>15.2f}")
