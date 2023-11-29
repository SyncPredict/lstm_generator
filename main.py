# Импорты
from components.data.process_data import create_data
from components.models.create_model import create_lstm_model, create_and_train_lstm_models
from components.models.train_and_evaluate_model import train_model
from components.models.read_lstm_configs import read_lstm_configs
from components.models.save_model_results import save_model_results
from components.models.compare_models import test_and_compare_models
import logging

# Настройка логирования
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')

if __name__ == "__main__":
    # Загрузка и предобработка данных
    X_train, y_train, X_val, y_val, X_test, y_test, test_dates = create_data("data.json")


    # Чтение конфигураций LSTM сетей
    lstm_configs = read_lstm_configs("lstm_configs.json")

    # Создание и обучение LSTM сетей
    create_and_train_lstm_models(lstm_configs, X_train, y_train, X_val, y_val)

    # Сравнительный анализ результатов
    test_and_compare_models(lstm_configs, X_test, y_test, test_dates)
