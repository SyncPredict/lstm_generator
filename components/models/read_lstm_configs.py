import json

def read_lstm_configs(config_path):
    """
    Чтение конфигураций LSTM сетей из файла JSON.

    :param config_path: Путь к файлу конфигурации.
    :return: Список конфигураций моделей.
    """
    with open(config_path, 'r') as file:

        configs = json.load(file)
    return configs['models'][:1]
