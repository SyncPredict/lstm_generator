import json
import pandas as pd
from sklearn.preprocessing import MinMaxScaler
from sklearn.model_selection import TimeSeriesSplit
import numpy as np

def create_sequences(data, n_steps):
    X, y = [], []
    for i in range(len(data) - n_steps):
        seq_x = data.iloc[i:i+n_steps].drop(['future_rate_change_1h'], axis=1) # Удаляем целевую переменную из X
        seq_y = data.iloc[i+n_steps]['future_rate_change_1h'] # Используем future_rate_change_1h в качестве y
        X.append(seq_x.values)
        y.append(seq_y)
    return np.array(X), np.array(y)

def create_data(file_path, future_interval):
    with open(file_path, 'r') as file:
        data = json.load(file)
    df = pd.DataFrame(data)

    df['date'] = pd.to_datetime(df['date'], unit='ms')
    df.set_index('date', inplace=True)

    df = df.resample('5T').mean().interpolate()

    # Расчет будущего изменения курса
    df['future_rate'] = df['rate'].shift(-future_interval) # Смещение курса в будущее
    df['future_rate_change_1h'] = df['future_rate'] - df['rate'] # Расчет величины изменения

    # Нормализация вектора изменения до +1, 0, или -1
    df['future_rate_change_1h'] = np.sign(df['future_rate_change_1h'])

    df.drop(['future_rate'], axis=1, inplace=True) # Удаление временного столбца
    df.dropna(inplace=True) # Удаление строк с NaN значениями

    scaler = MinMaxScaler(feature_range=(0, 1))
    df_scaled = scaler.fit_transform(df)
    df_scaled = pd.DataFrame(df_scaled, columns=df.columns, index=df.index)

    n_steps = 5
    X, y = create_sequences(df_scaled, n_steps)

    tscv = TimeSeriesSplit(n_splits=5)
    last_test_index = None

    for train_index, test_index in tscv.split(X):
        X_train, X_test = X[train_index], X[test_index]
        y_train, y_test = y[train_index], y[test_index]
        last_test_index = test_index

    if last_test_index is None:
        raise ValueError("Не удалось установить last_test_index")

    test_dates = df.index[last_test_index][-len(X_test):]

    val_size = int(len(X_train) * 0.2)
    X_val, y_val = X_train[-val_size:], y_train[-val_size:]
    X_train, y_train = X_train[:-val_size], y_train[:-val_size]

    return X_train, y_train, X_val, y_val, X_test, y_test, test_dates
