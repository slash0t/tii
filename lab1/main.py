import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.cluster import KMeans, AgglomerativeClustering
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import silhouette_score, calinski_harabasz_score, davies_bouldin_score
from scipy.cluster.hierarchy import dendrogram, linkage
from sklearn.manifold import MDS

# Настройка matplotlib
plt.rcParams['figure.figsize'] = (12, 8)
plt.rcParams['font.size'] = 11
plt.rcParams['axes.titlesize'] = 14
plt.rcParams['axes.labelsize'] = 12
plt.rcParams['xtick.labelsize'] = 10
plt.rcParams['ytick.labelsize'] = 10
plt.rcParams['legend.fontsize'] = 10

# Загружаем данные
data = pd.read_csv('digit.dat', sep=';')

# Подготавливаем данные
clean_data = data[['A', 'B', 'C', 'D', 'E', 'F', 'G', 'H']].copy()
clean_data.columns = ['DIGIT', 'VAR1', 'VAR2', 'VAR3', 'VAR4', 'VAR5', 'VAR6', 'VAR7']

# Очищаем данные
clean_data['DIGIT'] = clean_data['DIGIT'].str.strip()
for col in ['VAR1', 'VAR2', 'VAR3', 'VAR4', 'VAR5', 'VAR6', 'VAR7']:
    clean_data[col] = clean_data[col].str.strip().map({'ONE': 1, 'ZERO': 0})

# Подготавливаем данные для кластеризации
X = clean_data[['VAR1', 'VAR2', 'VAR3', 'VAR4', 'VAR5', 'VAR6', 'VAR7']].copy()
y = clean_data['DIGIT'].copy()

print(f"Распределение цифр:")
print(y.value_counts().sort_index())

# Стандартизация данных
scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)
X_scaled_df = pd.DataFrame(X_scaled, columns=X.columns)

# Вычисляем linkage matrix для разных методов
methods = ['ward']
linkage_matrices = linkage(X_scaled, method='ward')

# Строим дендрограмму для метода Ward
last = 30
plt.figure(figsize=(15, 10))
dendrogram(linkage_matrices,
           truncate_mode='lastp',  # показываем только последние p слияний
           p=last,  # количество последних слияний
           leaf_rotation=90,
           leaf_font_size=10,
           show_leaf_counts=True)
plt.title(f'Дендрограмма иерархической кластеризации (метод Ward)\nПоследние {last} слияний', fontsize=16, pad=20)
plt.xlabel('Индекс кластера или (размер кластера)', fontsize=12)
plt.ylabel('Расстояние', fontsize=12)
plt.grid(True, alpha=0.3)

# Добавляем горизонтальную линию для отсечения на уровне 5 кластеров
max_d = np.sort(linkage_matrices[:, 2])[-5]  # расстояние для 5 кластеров
plt.axhline(y=max_d, color='red', linestyle='--', linewidth=2,
            label=f'Разрез для 5 кластеров (d={max_d:.2f})')
plt.legend()
plt.tight_layout()
plt.show()

# Метод локтя и другие метрики
k_range = range(2, 16)
inertias = []
silhouette_scores = []
calinski_scores = []
davies_bouldin_scores = []

for k in k_range:
    kmeans = KMeans(n_clusters=k, random_state=42, n_init=10)
    cluster_labels = kmeans.fit_predict(X_scaled)

    inertias.append(kmeans.inertia_)
    silhouette_scores.append(silhouette_score(X_scaled, cluster_labels))
    calinski_scores.append(calinski_harabasz_score(X_scaled, cluster_labels))
    davies_bouldin_scores.append(davies_bouldin_score(X_scaled, cluster_labels))

# Строим графики для определения оптимального количества кластеров
plt.suptitle('Определение оптимального количества кластеров', fontsize=16, y=0.98)
axes = plt.gca()

# График "локтя" (инерция)
axes.plot(k_range, inertias, 'bo-', linewidth=2, markersize=8)
axes.set_title('Метод "локтя" (Инерция)', fontsize=14)
axes.set_xlabel('Количество кластеров (K)')
axes.set_ylabel('Внутрикластерная сумма квадратов')
axes.grid(True, alpha=0.3)

# Выделяем возможный локоть
elbow_k = 5  # примерно здесь видим перегиб
axes.axvline(x=elbow_k, color='red', linestyle='--', alpha=0.7, label=f'Возможный локоть (K={elbow_k})')
axes.legend()

plt.tight_layout()
plt.show()

# Финальная кластеризация для 10 кластеров (тк 10 цифр)
optimal_k = 10

# K-means кластеризация
kmeans_final = KMeans(n_clusters=optimal_k, random_state=42, n_init=10)
kmeans_labels = kmeans_final.fit_predict(X_scaled)

# Иерархическая кластеризация
hierarchical_final = AgglomerativeClustering(n_clusters=optimal_k, linkage='ward')
hierarchical_labels = hierarchical_final.fit_predict(X_scaled)

# Добавляем результаты к данным
analysis_data = clean_data.copy()
analysis_data['KMeans_Cluster'] = kmeans_labels
analysis_data['Hierarchical_Cluster'] = hierarchical_labels

# Оценка качества
kmeans_silhouette = silhouette_score(X_scaled, kmeans_labels)
hier_silhouette = silhouette_score(X_scaled, hierarchical_labels)

print(f"\nМетрики качества для K={optimal_k}:")
print(f"К-средних:")
print(f"  Silhouette Score: {kmeans_silhouette:.3f}")
print(f"Hierarchical:")
print(f"  Silhouette Score: {hier_silhouette:.3f}")

mds = MDS(n_components=2, random_state=42, dissimilarity='euclidean')
X_mds = mds.fit_transform(X_scaled)

# Создаем визуализацию результатов кластеризации
fig = plt.figure(figsize=(20, 15))

# Подготавливаем данные для визуализации
mds_data = pd.DataFrame({
    'MDS1': X_mds[:, 0],
    'MDS2': X_mds[:, 1],
    'DIGIT': analysis_data['DIGIT'],
    'KMeans_Cluster': analysis_data['KMeans_Cluster'],
    'Hierarchical_Cluster': analysis_data['Hierarchical_Cluster']
})

# Цветовая палитра для 10 кластеров
colors = plt.cm.tab10(np.linspace(0, 1, 10))

# Результат К-средних кластеризации
ax1 = plt.subplot(2, 3, 2)
scatter = ax1.scatter(X_mds[:, 0], X_mds[:, 1],
                      c=kmeans_labels, cmap='tab10',
                      alpha=0.7, s=50, edgecolors='black', linewidth=0.5)
ax1.set_title('K-means кластеризация (K=10)', fontsize=14, pad=10)
ax1.set_xlabel('MDS1')
ax1.set_ylabel('MDS2')
ax1.grid(True, alpha=0.3)
plt.colorbar(scatter, ax=ax1, label='Кластер')

# Результат иерархической кластеризации
ax2 = plt.subplot(2, 3, 1)
scatter2 = ax2.scatter(X_mds[:, 0], X_mds[:, 1],
                       c=hierarchical_labels, cmap='tab10',
                       alpha=0.7, s=50, edgecolors='black', linewidth=0.5)
ax2.set_title('Иерархическая кластеризация (K=10)', fontsize=14, pad=10)
ax2.set_xlabel('MDS1')
ax2.set_ylabel('MDS2')
ax2.grid(True, alpha=0.3)
plt.colorbar(scatter2, ax=ax2, label='Кластер')

# Истинные цифры
ax3 = plt.subplot(2, 3, 3)
digit_mapping = {'zero': 0, 'one': 1, 'two': 2, 'three': 3, 'four': 4,
                 'five': 5, 'six': 6, 'seven': 7, 'eight': 8, 'nine': 9}
true_labels_numeric = [digit_mapping[d] for d in analysis_data['DIGIT']]

scatter3 = ax3.scatter(X_mds[:, 0], X_mds[:, 1],
                       c=true_labels_numeric, cmap='tab10',
                       alpha=0.7, s=50, edgecolors='black', linewidth=0.5)
ax3.set_title('Истинные цифры', fontsize=14, pad=10)
ax3.set_xlabel('MDS1')
ax3.set_ylabel('MDS2')
ax3.grid(True, alpha=0.3)
cbar3 = plt.colorbar(scatter3, ax=ax3, label='Цифра')
cbar3.set_ticks(range(10))
cbar3.set_ticklabels(['0', '1', '2', '3', '4', '5', '6', '7', '8', '9'])

# Размеры кластеров иерархичксих
ax4 = plt.subplot(2, 3, 4)
cluster_sizes_hier = pd.Series(hierarchical_labels).value_counts().sort_index()
bars = ax4.bar(cluster_sizes_hier.index, cluster_sizes_hier.values,
               color=[colors[i] for i in cluster_sizes_hier.index], alpha=0.7, edgecolor='black')
ax4.set_title('Размеры кластеров (Иерархичксий)', fontsize=14, pad=10)
ax4.set_xlabel('Номер кластера')
ax4.set_ylabel('Количество объектов')
ax4.grid(True, alpha=0.3, axis='y')

for bar in bars:
    height = bar.get_height()
    ax4.text(bar.get_x() + bar.get_width() / 2., height + 0.5,
             f'{int(height)}', ha='center', va='bottom', fontsize=10)

# Размеры кластеров ксредних
ax5 = plt.subplot(2, 3, 5)
cluster_sizes_kmeans = pd.Series(kmeans_labels).value_counts().sort_index()
bars = ax5.bar(cluster_sizes_kmeans.index, cluster_sizes_kmeans.values,
               color=[colors[i] for i in cluster_sizes_kmeans.index], alpha=0.7, edgecolor='black')
ax5.set_title('Размеры кластеров (K-means)', fontsize=14, pad=10)
ax5.set_xlabel('Номер кластера')
ax5.set_ylabel('Количество объектов')
ax5.grid(True, alpha=0.3, axis='y')

for bar in bars:
    height = bar.get_height()
    ax5.text(bar.get_x() + bar.get_width() / 2., height + 0.5,
             f'{int(height)}', ha='center', va='bottom', fontsize=10)

# 6. Сравнение методов
ax6 = plt.subplot(2, 3, 6)
metrics_comparison = pd.DataFrame({
    'K-means': [kmeans_silhouette],
    'Hierarchical': [hier_silhouette]
}, index=['Silhouette'])

metrics_comparison.plot(kind='bar', ax=ax6, color=['skyblue', 'lightcoral'], alpha=0.8)
ax6.set_title('Сравнение методов кластеризации', fontsize=14, pad=10)
ax6.set_ylabel('Нормализованные значения')
ax6.set_xticklabels(ax6.get_xticklabels(), rotation=45, ha='right')
ax6.grid(True, alpha=0.3, axis='y')
ax6.legend()

plt.tight_layout()
plt.show()

# Анализ центров кластеров
centers = kmeans_final.cluster_centers_
centers_original = scaler.inverse_transform(centers)
centers_df = pd.DataFrame(centers_original,
                          columns=['VAR1', 'VAR2', 'VAR3', 'VAR4', 'VAR5', 'VAR6', 'VAR7'])
centers_df.index.name = 'Кластер'

print("Центры кластеров (исходный масштаб 0-1):")
print(centers_df.round(3))

# Анализируем соответствие кластеров цифрам
print("\nДоминирующие цифры в каждом кластере:")
cluster_interpretations = {}
for cluster in range(optimal_k):
    cluster_data = analysis_data[analysis_data['KMeans_Cluster'] == cluster]
    if len(cluster_data) > 0:
        dominant_digit = cluster_data['DIGIT'].value_counts().index[0]
        count = cluster_data['DIGIT'].value_counts().iloc[0]
        total = len(cluster_data)
        percentage = (count / total) * 100
        cluster_interpretations[cluster] = dominant_digit
        print(f"Кластер {cluster}: '{dominant_digit}' ({count}/{total} = {percentage:.1f}%)")

def cluster_purity(true_labels, cluster_labels):
    total = len(true_labels)
    purity_sum = 0

    for cluster in np.unique(cluster_labels):
        cluster_mask = cluster_labels == cluster
        if np.sum(cluster_mask) > 0:
            cluster_true_labels = true_labels[cluster_mask]
            most_common = pd.Series(cluster_true_labels).value_counts().iloc[0]
            purity_sum += most_common

    return purity_sum / total


# Конвертируем цифры в числовой формат
true_labels_array = np.array([digit_mapping[d] for d in analysis_data['DIGIT']])

kmeans_purity = cluster_purity(true_labels_array, kmeans_labels)
hierarchical_purity = cluster_purity(true_labels_array, hierarchical_labels)

print(f"Чистота кластеризации К-средних: {kmeans_purity:.3f}")
print(f"Чистота кластеризации иерархической: {hierarchical_purity:.3f}")

print(f"Выполнена кластеризация двумя методами с K={optimal_k}")
print(f"К-средних показал лучшие результаты по всем метрикам:")
print(f"   - Silhouette Score: {kmeans_silhouette:.3f}")
print(f"   - Чистота кластеров: {kmeans_purity:.1%}")

print(f"\n Кластера и их соответствие цифрам:")
digit_cluster_map = {}
for cluster in range(optimal_k):
    cluster_data = analysis_data[analysis_data['KMeans_Cluster'] == cluster]
    if len(cluster_data) > 0:
        dominant_digit = cluster_data['DIGIT'].value_counts().index[0]
        purity = cluster_data['DIGIT'].value_counts().iloc[0] / len(cluster_data)
        digit_cluster_map[dominant_digit] = (cluster, purity)
        print(f"   Цифра '{dominant_digit}' -> Кластер {cluster} (чистота: {purity:.1%})")
