
from keras.models import Sequential
from keras.layers import LSTM, Dense, Dropout, Bidirectional, GRU, Activation, BatchNormalization, TimeDistributed, \
    SimpleRNN
from keras.optimizers import Adam, RMSprop, SGD

from components.models.save_model_results import save_model_results
from components.models.train_and_evaluate_model import train_model


def create_lstm_model(config):
    """
    Создает LSTM модель на основе заданной конфигурации.

    :param config:
    :return: Скомпилированная модель.
    """

    model = Sequential()

    for i, layer in enumerate(config['layers']):
        # Добавление разных типов слоев
        if layer['type'] == 'LSTM':
            lstm_layer = LSTM(layer['units'], return_sequences=(i < len(config['layers']) - 1))
        elif layer['type'] == 'BiLSTM':
            lstm_layer = Bidirectional(LSTM(layer['units'], return_sequences=(i < len(config['layers']) - 1)))
        elif layer['type'] == 'GRU':
            lstm_layer = GRU(layer['units'], return_sequences=(i < len(config['layers']) - 1))
        elif layer['type'] == 'SimpleRNN':
            lstm_layer = SimpleRNN(layer['units'], return_sequences=(i < len(config['layers']) - 1))
        else:
            raise ValueError(f"Unsupported layer type: {layer['type']}")

        model.add(lstm_layer)

        # Добавление Dropout, если указано
        if 'dropout' in layer:
            model.add(Dropout(layer['dropout']))

        # Добавление BatchNormalization, если указано
        if layer.get('batch_normalization', False):
            model.add(BatchNormalization())

        # Добавление TimeDistributed, если указано
        if layer.get('time_distributed', False):
            model.add(TimeDistributed(Dense(layer['units'])))

    # Выходной слой
    model.add(Dense(config['output_units']))
    model.add(Activation(config['output_activation']))

    # Выбор оптимизатора
    if config['optimizer'] == 'adam':
        optimizer = Adam(learning_rate=config['learning_rate'])
    elif config['optimizer'] == 'rmsprop':
        optimizer = RMSprop(learning_rate=config['learning_rate'])
    elif config['optimizer'] == 'sgd':
        optimizer = SGD(learning_rate=config['learning_rate'])
    else:
        raise ValueError(f"Unsupported optimizer: {config['optimizer']}")

    model.compile(optimizer=optimizer, loss=config['loss'], metrics=config['metrics'])

    return model




def create_and_train_lstm_models(lstm_configs,X_train, y_train, X_val, y_val):
    for config in lstm_configs:
        # Создание и компиляция модели
        model = create_lstm_model(config)

        # Обучение и оценка модели
        train_config = {
            'epochs': 50,
            'batch_size': 64,
            'patience': 10
        }

        history = train_model(model, X_train, y_train, X_val, y_val, train_config, config['name'])

        # Сохранение результатов
        save_model_results(model, history, config['name'])  # Убедитесь, что конфигурация содержит 'name'
