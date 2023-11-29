import matplotlib.pyplot as plt
import pandas as pd
from tensorflow.keras.models import load_model
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score
import numpy as np


def test_and_compare_models(lstm_configs, X_test, y_test, test_dates):
    print(y_test)
    """
    Сравнивает различные LSTM модели для прогнозирования цены биткоина.
    """
    results = []
    plt.figure(figsize=(15, 10))

    for config in lstm_configs:
        # Загрузка модели
        model_name = config['name']
        # model = load_model(f"models/{model_name}.h5")
        model = load_model(f"models/best_{model_name}.h5")

        # Прогнозирование
        y_pred = model.predict(X_test)

        # Вычисление метрик
        mse = mean_squared_error(y_test, y_pred)
        mae = mean_absolute_error(y_test, y_pred)
        rmse = np.sqrt(mse)
        r2 = r2_score(y_test, y_pred)

        results.append({
            'model': model_name,
            'MSE': mse,
            'MAE': mae,
            'RMSE': rmse,
            'R2': r2
        })

        # Визуализация
        plt.plot(test_dates, y_pred, label=f'Predicted by {model_name}')

    # Отображение реальных данных
    plt.plot(test_dates,y_test, label='Real', color='black', linewidth=2.0)

    plt.title('Real vs Predicted Bitcoin Prices')
    plt.xlabel('Time')
    plt.ylabel('Bitcoin Price')
    plt.legend()
    plt.grid(True)
    plt.xticks(rotation=45)  # Поворачивает метки на оси X для лучшей читаемости
    plt.tight_layout()
    plt.savefig(f'plots/plot.png', dpi=300)
    plt.show()

    # Вывод результатов
    results_df = pd.DataFrame(results)
    print(results_df)

    return results_df

# Дополнительно можно рассмотреть:
# 1. Использование разных подходов для нормализации данных.
# 2. Проведение анализа влияния различных гиперпараметров на результаты.
# 3. Разработка ансамблевых моделей для повышения точности.
