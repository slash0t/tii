"""
=============================================================================
ЛАБОРАТОРНАЯ РАБОТА: ЛИНЕЙНЫЙ РЕГРЕССИОННЫЙ АНАЛИЗ И ПРОГНОЗИРОВАНИЕ
=============================================================================

Тема: Прогнозирование средней цены в долларах по Chicago Metropolitan Area
      Hotel Statistics (переменная x4)

Цель: Построить прогноз на 8 месяцев средней цены в долларах

Автор: [Ваше имя]
Дата: 2025
=============================================================================
"""

import warnings

warnings.filterwarnings("ignore")

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from statsmodels.tsa.seasonal import seasonal_decompose
import matplotlib.dates as mdates
from scipy import stats

# Настройки для красивых графиков
plt.style.use('seaborn-v0_8-darkgrid')
plt.rcParams['figure.figsize'] = (14, 6)
plt.rcParams['font.size'] = 10

print("=" * 80)
print("ЛИНЕЙНЫЙ РЕГРЕССИОННЫЙ АНАЛИЗ: ПРОГНОЗИРОВАНИЕ ВРЕМЕННЫХ РЯДОВ")
print("=" * 80)
print("\nЦель: Прогноз средней цены отелей Chicago Metropolitan Area на 8 месяцев")
print("-" * 80)

"""
=============================================================================
ТЕОРЕТИЧЕСКАЯ ЧАСТЬ
=============================================================================

1. ВРЕМЕННОЙ РЯД (Time Series)
   Последовательность наблюдений, упорядоченных во времени.
   Основные компоненты:
   - Тренд (Trend): долгосрочная тенденция изменения
   - Сезонность (Seasonality): регулярные колебания с фиксированным периодом
   - Циклы (Cycles): нерегулярные долгосрочные колебания
   - Случайная компонента (Residual): нерегулярные остатки

2. АДДИТИВНАЯ МОДЕЛЬ
   Y(t) = T(t) + S(t) + R(t)
   где T - тренд, S - сезонность, R - остатки
   Используется когда амплитуда сезонных колебаний постоянна

3. МУЛЬТИПЛИКАТИВНАЯ МОДЕЛЬ
   Y(t) = T(t) * S(t) * R(t)
   Используется когда амплитуда сезонных колебаний растет с трендом

4. ЛИНЕЙНАЯ РЕГРЕССИЯ ДЛЯ ТРЕНДА
   y = a*x + b (полиномиальная: y = a_n*x^n + ... + a_1*x + a_0)
   Метод наименьших квадратов минимизирует сумму квадратов отклонений

5. ПРОГНОЗИРОВАНИЕ
   Экстраполяция = продолжение найденных закономерностей в будущее
   Точность снижается с увеличением горизонта прогноза
=============================================================================
"""

# =============================================================================
# ШАГ 1: ЗАГРУЗКА И ПРЕДВАРИТЕЛЬНАЯ ОБРАБОТКА ДАННЫХ
# =============================================================================
print("\n" + "=" * 80)
print("ШАГ 1: ЗАГРУЗКА И ПРЕДВАРИТЕЛЬНАЯ ОБРАБОТКА ДАННЫХ")
print("=" * 80)

FILENAME = "Chicago_hotels.csv"  # Укажите правильный путь к файлу
FORECAST_MONTHS = 8
SEASONAL_PERIOD = 12  # Годовая сезонность (12 месяцев)

print(f"\nЗагрузка данных из файла: {FILENAME}")
print(f"Целевая переменная: x4 (Average Daily Rate, Chicago Metropolitan Area)")
print(f"Горизонт прогноза: {FORECAST_MONTHS} месяцев")

# Чтение данных
df = pd.read_csv(FILENAME, sep=';', decimal=',', dtype=str)

# Очистка пробелов
df = df.applymap(lambda x: x.strip() if isinstance(x, str) else x)

# Создание даты из месяца и года
df['month_year'] = df['date1'].astype(str) + '-' + df['date2'].astype(str)
df['date'] = pd.to_datetime(df['month_year'], format='%B-%Y', errors='coerce')

# Проверка на некорректные даты
bad_dates = df[df['date'].isna()]
if len(bad_dates) > 0:
    print(f"\n⚠️  Найдено {len(bad_dates)} строк с некорректными датами (будут удалены)")
    df = df.dropna(subset=['date'])

# Преобразование x4 в числовой формат
df['x4'] = df['x4'].str.replace(',', '.').astype(float)

# Фильтрация: оставляем только дату и x4
df = df[['date', 'x4']].copy()
df = df.sort_values('date').set_index('date')

# Удаление дубликатов по дате
df = df[~df.index.duplicated(keep='first')]

# Удаление строк с пропущенными значениями x4
df = df.dropna(subset=['x4'])

print(f"\n✓ Данные успешно загружены")
print(f"  Период данных: {df.index.min().strftime('%B %Y')} - {df.index.max().strftime('%B %Y')}")
print(f"  Количество наблюдений: {len(df)}")
print(f"  Диапазон значений: ${df['x4'].min():.2f} - ${df['x4'].max():.2f}")

series = df['x4']

# =============================================================================
# ШАГ 2: ВИЗУАЛИЗАЦИЯ ИСХОДНОГО ВРЕМЕННОГО РЯДА
# =============================================================================
print("\n" + "=" * 80)
print("ШАГ 2: ПОСТРОЕНИЕ ГРАФИКА ИСХОДНОГО ВРЕМЕННОГО РЯДА")
print("=" * 80)

plt.figure(figsize=(14, 6))
plt.plot(series.index, series.values, marker='o', markersize=4,
         linewidth=1.5, color='steelblue', label='Исходные данные')
plt.title('Исходный временной ряд: Average Daily Rate ($)\nChicago Metropolitan Area Hotel Statistics',
          fontsize=14, fontweight='bold')
plt.xlabel('Дата', fontsize=12)
plt.ylabel('Средняя цена ($)', fontsize=12)
plt.grid(True, alpha=0.3)
plt.legend(fontsize=11)

# Форматирование оси X
ax = plt.gca()
ax.xaxis.set_major_formatter(mdates.DateFormatter('%b %Y'))
ax.xaxis.set_major_locator(mdates.YearLocator())
plt.xticks(rotation=45, ha='right')
plt.tight_layout()
plt.show()

print("\n📊 График исходного ряда построен")

# =============================================================================
# ШАГ 3: АНАЛИЗ ХАРАКТЕРИСТИК ВРЕМЕННОГО РЯДА
# =============================================================================
print("\n" + "=" * 80)
print("ШАГ 3: ОТВЕТЫ НА ВОПРОСЫ О ХАРАКТЕРИСТИКАХ РЯДА")
print("=" * 80)

# Вопрос 1: Есть ли у ряда тренд?
print("\n1️⃣  АНАЛИЗ ТРЕНДА")
print("-" * 40)

# Линейная регрессия для оценки тренда
x = np.arange(len(series))
y = series.values
slope, intercept, r_value, p_value, std_err = stats.linregress(x, y)

print(f"   Коэффициент наклона: {slope:.4f}")
print(f"   R² (коэффициент детерминации): {r_value ** 2:.4f}")
print(f"   p-value: {p_value:.6f}")

if p_value < 0.05 and abs(r_value) > 0.3:
    trend_direction = "восходящий" if slope > 0 else "нисходящий"
    print(f"\n   ✓ Да, у ряда есть СТАТИСТИЧЕСКИ ЗНАЧИМЫЙ {trend_direction} тренд")
    print(f"     Средний рост: ${slope:.4f} за месяц")
else:
    print("\n   ✓ Явного линейного тренда не обнаружено")

# Вопрос 2: Есть ли сезонность?
print("\n2️⃣  АНАЛИЗ СЕЗОННОСТИ")
print("-" * 40)

# Декомпозиция для выявления сезонности
decomp = seasonal_decompose(series, model='additive', period=SEASONAL_PERIOD,
                            extrapolate_trend='freq')

seasonal_strength = 1 - (decomp.resid.var() / (decomp.seasonal + decomp.resid).var())
print(f"   Сила сезонной компоненты: {seasonal_strength:.4f}")

if seasonal_strength > 0.3:
    print(f"\n   ✓ Да, присутствует ВЫРАЖЕННАЯ сезонность с периодом {SEASONAL_PERIOD} месяцев")
    print(f"     Это годовая сезонность (12 месяцев)")

    # Анализ сезонного паттерна
    seasonal_pattern = decomp.seasonal[:SEASONAL_PERIOD]
    max_month = seasonal_pattern.idxmax().month
    min_month = seasonal_pattern.idxmin().month
    month_names = ['', 'Январь', 'Февраль', 'Март', 'Апрель', 'Май', 'Июнь',
                   'Июль', 'Август', 'Сентябрь', 'Октябрь', 'Ноябрь', 'Декабрь']
    print(f"     Пик цен: {month_names[max_month]} (+${seasonal_pattern.max():.2f})")
    print(f"     Минимум цен: {month_names[min_month]} (${seasonal_pattern.min():.2f})")
else:
    print("\n   ✓ Сезонность слабая или отсутствует")

# Вопрос 3: Меняет ли ряд свой характер?
print("\n3️⃣  АНАЛИЗ ИЗМЕНЕНИЯ ХАРАКТЕРА РЯДА")
print("-" * 40)

# Разбиваем ряд на части и сравниваем
mid_point = len(series) // 2
first_half = series[:mid_point]
second_half = series[mid_point:]

mean_diff = abs(second_half.mean() - first_half.mean())
std_diff = abs(second_half.std() - first_half.std())

print(f"   Средняя цена (1-я половина): ${first_half.mean():.2f}")
print(f"   Средняя цена (2-я половина): ${second_half.mean():.2f}")
print(f"   Изменение среднего: ${mean_diff:.2f}")
print(f"   Изменение стандартного отклонения: ${std_diff:.2f}")

if mean_diff > first_half.std() * 0.5:
    print(f"\n   ✓ Да, ряд МЕНЯЕТ свой характер: наблюдается сдвиг уровня")
else:
    print(f"\n   ✓ Характер ряда относительно стабилен")

# Вопрос 4: Есть ли выбросы?
print("\n4️⃣  АНАЛИЗ ВЫБРОСОВ")
print("-" * 40)

# Метод межквартильного размаха (IQR)
q1, q3 = series.quantile([0.25, 0.75])
iqr = q3 - q1
lower_bound = q1 - 1.5 * iqr
upper_bound = q3 + 1.5 * iqr
outliers = series[(series < lower_bound) | (series > upper_bound)]

print(f"   Нижняя граница (Q1 - 1.5*IQR): ${lower_bound:.2f}")
print(f"   Верхняя граница (Q3 + 1.5*IQR): ${upper_bound:.2f}")
print(f"   Найдено выбросов: {len(outliers)}")

if len(outliers) > 0:
    print(f"\n   ✓ Обнаружены выбросы:")
    for date, value in outliers.items():
        print(f"     - {date.strftime('%B %Y')}: ${value:.2f}")
else:
    print(f"\n   ✓ Выбросов не обнаружено")

# =============================================================================
# ШАГ 4: ДЕКОМПОЗИЦИЯ ВРЕМЕННОГО РЯДА
# =============================================================================
print("\n" + "=" * 80)
print("ШАГ 4: ДЕКОМПОЗИЦИЯ ВРЕМЕННОГО РЯДА")
print("=" * 80)
print("\nДекомпозиция разделяет ряд на компоненты: Тренд + Сезонность + Остатки")

fig = decomp.plot()
fig.set_size_inches(14, 10)

# Улучшаем форматирование графиков
for i, ax in enumerate(fig.axes):
    ax.xaxis.set_major_formatter(mdates.DateFormatter('%b %Y'))
    ax.xaxis.set_major_locator(mdates.YearLocator())
    plt.setp(ax.xaxis.get_majorticklabels(), rotation=45, ha="right")
    ax.grid(True, alpha=0.3)

    # Добавляем описания
    titles = [
        'Исходный ряд (Observed)',
        'Тренд (Trend) - долгосрочная тенденция',
        'Сезонность (Seasonal) - годовой цикл',
        'Остатки (Residual) - случайная компонента'
    ]
    if i < len(titles):
        ax.set_title(titles[i], fontsize=11, fontweight='bold')

plt.suptitle("Декомпозиция временного ряда (Аддитивная модель)",
             fontsize=14, fontweight='bold', y=0.995)
plt.tight_layout()
plt.show()

print("\n📊 Декомпозиция выполнена")

# =============================================================================
# ШАГ 5: ВИЗУАЛИЗАЦИЯ ВЫБРОСОВ
# =============================================================================
print("\n" + "=" * 80)
print("ШАГ 5: ВИЗУАЛИЗАЦИЯ ВЫБРОСОВ")
print("=" * 80)

plt.figure(figsize=(14, 6))
plt.plot(series.index, series.values, marker='o', markersize=4,
         linewidth=1.5, color='steelblue', label='Исходные данные')

if len(outliers) > 0:
    plt.scatter(outliers.index, outliers.values, color='red', s=100,
                zorder=5, label=f'Выбросы (n={len(outliers)})', marker='X')

plt.axhline(y=upper_bound, color='orange', linestyle='--',
            linewidth=1, label=f'Верхняя граница (${upper_bound:.2f})')
plt.axhline(y=lower_bound, color='orange', linestyle='--',
            linewidth=1, label=f'Нижняя граница (${lower_bound:.2f})')

plt.title('Исходный ряд с выделенными выбросами (метод IQR)',
          fontsize=14, fontweight='bold')
plt.xlabel('Дата', fontsize=12)
plt.ylabel('Средняя цена ($)', fontsize=12)
plt.grid(True, alpha=0.3)
plt.legend(fontsize=10)

ax = plt.gca()
ax.xaxis.set_major_formatter(mdates.DateFormatter('%b %Y'))
ax.xaxis.set_major_locator(mdates.YearLocator())
plt.xticks(rotation=45, ha='right')
plt.tight_layout()
plt.show()

print("\n📊 График с выбросами построен")

# =============================================================================
# ШАГ 6: ПОСТРОЕНИЕ ЛИНЕЙНОЙ РЕГРЕССИОННОЙ МОДЕЛИ
# =============================================================================
print("\n" + "=" * 80)
print("ШАГ 6: ПОСТРОЕНИЕ ЛИНЕЙНОЙ РЕГРЕССИОННОЙ МОДЕЛИ")
print("=" * 80)

print("\nИспользуем полиномиальную регрессию для моделирования тренда")
print("Порядок полинома: 3 (баланс между простотой и точностью)")

# Извлекаем тренд из декомпозиции
trend = decomp.trend.dropna()
x_trend = np.arange(len(trend))
y_trend = trend.values

# Полиномиальная регрессия 3-й степени
poly_degree = 3
coeffs = np.polyfit(x_trend, y_trend, poly_degree)
trend_func = np.poly1d(coeffs)

print(f"\nКоэффициенты полинома (степень {poly_degree}):")
for i, coef in enumerate(coeffs):
    print(f"  a_{poly_degree - i} = {coef:.6f}")

# Оценка качества модели
trend_fitted = trend_func(x_trend)
r2 = 1 - np.sum((y_trend - trend_fitted) ** 2) / np.sum((y_trend - np.mean(y_trend)) ** 2)
rmse = np.sqrt(np.mean((y_trend - trend_fitted) ** 2))

print(f"\nКачество модели:")
print(f"  R² = {r2:.4f} (доля объясненной дисперсии)")
print(f"  RMSE = ${rmse:.2f} (средняя ошибка)")

# =============================================================================
# ШАГ 7: ПРОГНОЗИРОВАНИЕ
# =============================================================================
print("\n" + "=" * 80)
print("ШАГ 7: ВЫПОЛНЕНИЕ ПРОГНОЗА НА 8 МЕСЯЦЕВ")
print("=" * 80)

print("\nМетод прогнозирования: Декомпозиция + Полиномиальная экстраполяция")
print("Прогноз = Тренд (экстраполированный) + Сезонность (повторяющийся паттерн)")

# Прогноз тренда
x_future = np.arange(len(trend), len(trend) + FORECAST_MONTHS)
trend_forecast = trend_func(x_future)

print(f"\nПрогноз тренда на {FORECAST_MONTHS} месяцев:")
for i, val in enumerate(trend_forecast, 1):
    print(f"  Месяц +{i}: ${val:.2f}")

# Сезонность
seasonal = decomp.seasonal
season_pattern = seasonal[:SEASONAL_PERIOD].values
season_forecast = [season_pattern[i % SEASONAL_PERIOD] for i in range(FORECAST_MONTHS)]

print(f"\nСезонная корректировка:")
for i, val in enumerate(season_forecast, 1):
    print(f"  Месяц +{i}: {val:+.2f}")

# Итоговый прогноз
forecast = trend_forecast + season_forecast

print(f"\n{'=' * 60}")
print("ИТОГОВЫЙ ПРОГНОЗ (Тренд + Сезонность):")
print(f"{'=' * 60}")

# Создание дат для прогноза
last_date = series.index[-1]
future_dates = pd.date_range(last_date + pd.DateOffset(months=1),
                             periods=FORECAST_MONTHS, freq='MS')
forecast_series = pd.Series(forecast, index=future_dates)

for date, value in forecast_series.items():
    print(f"  {date.strftime('%Y-%m-%d')} ({date.strftime('%B %Y')}): ${value:.2f}")

# =============================================================================
# ШАГ 8: ВИЗУАЛИЗАЦИЯ РЕЗУЛЬТАТОВ И ПРОГНОЗА
# =============================================================================
print("\n" + "=" * 80)
print("ШАГ 8: ПОСТРОЕНИЕ ИТОГОВЫХ ГРАФИКОВ")
print("=" * 80)

fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(14, 10))

# График 1: Исходные данные + Прогноз
ax1.plot(series.index, series.values, marker='o', markersize=4,
         linewidth=1.5, color='steelblue', label='Исходные данные (observed)')
ax1.plot(forecast_series.index, forecast_series.values, marker='s',
         markersize=6, linewidth=2, color='green',
         label=f'Прогноз на {FORECAST_MONTHS} месяцев')

# Доверительный интервал (примерный, на основе RMSE)
ci = 1.96 * rmse  # 95% доверительный интервал
ax1.fill_between(forecast_series.index,
                 forecast_series.values - ci,
                 forecast_series.values + ci,
                 alpha=0.2, color='green',
                 label='95% доверительный интервал')

ax1.axvline(x=last_date, color='red', linestyle='--',
            linewidth=1.5, label='Граница прогноза')
ax1.set_title('Исходный временной ряд и прогноз на 8 месяцев',
              fontsize=13, fontweight='bold')
ax1.set_xlabel('Дата', fontsize=11)
ax1.set_ylabel('Средняя цена ($)', fontsize=11)
ax1.grid(True, alpha=0.3)
ax1.legend(fontsize=10, loc='best')
ax1.xaxis.set_major_formatter(mdates.DateFormatter('%b %Y'))
ax1.xaxis.set_major_locator(mdates.YearLocator())
plt.setp(ax1.xaxis.get_majorticklabels(), rotation=45, ha="right")

# График 2: Декомпозиция с прогнозом
trend_full = pd.concat([trend, pd.Series(trend_forecast, index=future_dates)])
seasonal_full = pd.concat([seasonal, pd.Series(season_forecast, index=future_dates)])

ax2.plot(series.index, series.values, linewidth=1.5,
         color='steelblue', label='Исходные данные', alpha=0.7)
ax2.plot(trend_full.index, trend_full.values, linewidth=2,
         color='red', linestyle='--', label='Тренд (экстраполированный)')
ax2.plot(forecast_series.index, forecast_series.values,
         linewidth=2.5, color='green', marker='s',
         markersize=6, label='Прогноз (тренд + сезонность)')
ax2.axvline(x=last_date, color='red', linestyle='--',
            linewidth=1.5, alpha=0.5)

ax2.set_title('Декомпозиция: Тренд и прогноз',
              fontsize=13, fontweight='bold')
ax2.set_xlabel('Дата', fontsize=11)
ax2.set_ylabel('Средняя цена ($)', fontsize=11)
ax2.grid(True, alpha=0.3)
ax2.legend(fontsize=10, loc='best')
ax2.xaxis.set_major_formatter(mdates.DateFormatter('%b %Y'))
ax2.xaxis.set_major_locator(mdates.YearLocator())
plt.setp(ax2.xaxis.get_majorticklabels(), rotation=45, ha="right")

plt.tight_layout()
plt.show()

print("\n📊 Итоговые графики построены")

# =============================================================================
# ШАГ 9: НАБОР СПРОГНОЗИРОВАННЫХ ЗНАЧЕНИЙ
# =============================================================================
print("\n" + "=" * 80)
print("ШАГ 9: НАБОР СПРОГНОЗИРОВАННЫХ ЗНАЧЕНИЙ")
print("=" * 80)

print("\nСПРОГНОЗИРОВАННЫЕ ЗНАЧЕНИЯ Average Daily Rate ($)")
print("Chicago Metropolitan Area Hotel Statistics")
print("-" * 60)

for i, (date, value) in enumerate(forecast_series.items(), 1):
    month_name = date.strftime('%B')
    year = date.strftime('%Y')
    print(f"{i}. {month_name:12} {year}: ${value:7.2f}")

print("-" * 60)
print(f"Средний прогнозируемый уровень: ${forecast_series.mean():.2f}")
print(f"Минимальное значение прогноза:  ${forecast_series.min():.2f}")
print(f"Максимальное значение прогноза: ${forecast_series.max():.2f}")

# =============================================================================
# ВЫВОДЫ
# =============================================================================
print("\n" + "=" * 80)
print("ВЫВОДЫ")
print("=" * 80)

print(f"""
1. ХАРАКТЕРИСТИКИ ВРЕМЕННОГО РЯДА:
   - Тренд: {"Обнаружен" if p_value < 0.05 else "Не выражен"} 
     ({trend_direction if p_value < 0.05 else "стабильный"})
   - Сезонность: {"Выраженная" if seasonal_strength > 0.3 else "Слабая"} 
     (период {SEASONAL_PERIOD} месяцев)
   - Выбросы: Обнаружено {len(outliers)} значений
   - Изменение характера: {"Да" if mean_diff > first_half.std() * 0.5 else "Нет"}

2. МОДЕЛЬ ПРОГНОЗИРОВАНИЯ:
   - Метод: Декомпозиция + Полиномиальная регрессия (степень {poly_degree})
   - Качество модели: R² = {r2:.4f}, RMSE = ${rmse:.2f}
   - Горизонт прогноза: {FORECAST_MONTHS} месяцев

3. РЕЗУЛЬТАТЫ ПРОГНОЗА:
   - Прогнозируемый диапазон: ${forecast_series.min():.2f} - ${forecast_series.max():.2f}
   - Средний уровень: ${forecast_series.mean():.2f}
   - Тенденция: {"Рост" if trend_forecast[-1] > trend_forecast[0] else "Снижение"} 
     средних цен

4. ОГРАНИЧЕНИЯ И РЕКОМЕНДАЦИИ:
   - Прогноз основан на исторических закономерностях и предполагает их сохранение
   - Не учитываются внешние факторы (экономические кризисы, пандемии и т.д.)
   - Точность снижается с увеличением горизонта прогноза
   - Рекомендуется регулярно обновлять модель с новыми данными
   - Для более точного прогноза можно использовать ARIMA, SARIMA или ML-методы
""")

print("=" * 80)
print("ЛАБОРАТОРНАЯ РАБОТА ЗАВЕРШЕНА")
print("=" * 80)

# Сохранение результатов в CSV
output_df = pd.DataFrame({
    'Дата': forecast_series.index,
    'Прогноз ($)': forecast_series.values
})
output_df.to_csv('forecast_results.csv', index=False)
print("\n💾 Результаты сохранены в файл: forecast_results.csv")