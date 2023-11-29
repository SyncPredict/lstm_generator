import json
import pandas as pd
from sklearn.preprocessing import MinMaxScaler
from sklearn.model_selection import TimeSeriesSplit
import numpy as np

def create_sequences(data, n_steps):
    X, y = [], []
    for i in range(len(data) - n_steps):
        # Включаем все признаки для X
        seq_x = data.iloc[i:i+n_steps]
        # Используем значение 'rate' следующего шага в качестве y
        seq_y = data.iloc[i+n_steps]['rate']
        X.append(seq_x.values)
        y.append(seq_y)
    return np.array(X), np.array(y)

def create_data(file_path):
    # Загрузка и подготовка данных
    with open(file_path, 'r') as file:
        data = json.load(file)
    df = pd.DataFrame(data)

    # Преобразование и индексация по времени
    df['date'] = pd.to_datetime(df['date'], unit='ms')
    df.set_index('date', inplace=True)

    # Линейная интерполяция для заполнения пропущенных значений
    df = df.resample('5T').mean().interpolate()

    # Нормализация
    scaler = MinMaxScaler(feature_range=(0, 1))
    df_scaled = scaler.fit_transform(df)
    df_scaled = pd.DataFrame(df_scaled, columns=df.columns, index=df.index)

    # Параметры последовательности
    n_steps = 5

    # Создание последовательностей
    X, y = create_sequences(df_scaled, n_steps)

    # Разделение данных
    tscv = TimeSeriesSplit(n_splits=5)
    last_test_index = None  # Инициализация переменной для сохранения последнего test_index

    for train_index, test_index in tscv.split(X):
        X_train, X_test = X[train_index], X[test_index]
        y_train, y_test = y[train_index], y[test_index]
        last_test_index = test_index  # Сохраняем test_index на каждой итерации

        # Проверяем, был ли last_test_index установлен
    if last_test_index is None:
        raise ValueError("Не удалось установить last_test_index")

        # Получение дат для тестового набора
    test_dates = df.index[last_test_index][-len(X_test):]

    # Разделение на обучающий и валидационный наборы
    val_size = int(len(X_train) * 0.2)
    X_val, y_val = X_train[-val_size:], y_train[-val_size:]
    X_train, y_train = X_train[:-val_size], y_train[:-val_size]

    return X_train, y_train, X_val, y_val, X_test, y_test, test_dates
