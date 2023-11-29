from keras.callbacks import EarlyStopping, ModelCheckpoint, TensorBoard
import os
import datetime
import numpy as np


def train_model(model, X_train, y_train, X_val, y_val, config, model_name):
    """
    Обучает LSTM модель.
    :param model: Модель для обучения.
    :param X_train: Обучающие данные.
    :param y_train: Обучающие метки.
    :param X_val: Валидационные данные.
    :param y_val: Валидационные метки.
    :param config: Конфигурация параметров обучения.
    :return: История обучения
    """
    # Настройка путей для сохранения модели и логов
    checkpoint_path = f'models/best_{model_name}.h5'
    log_dir = "logs/" + model_name + "/fit/" + datetime.datetime.now().strftime("%Y%m%d-%H%M%S")

    # Создание директорий, если они не существуют
    os.makedirs(os.path.dirname(checkpoint_path), exist_ok=True)
    os.makedirs(os.path.dirname(log_dir), exist_ok=True)

    # Коллбэки
    callbacks = [
        EarlyStopping(monitor='val_loss', patience=config['patience'], verbose=1),
        ModelCheckpoint(filepath=checkpoint_path, monitor='val_loss', save_best_only=True, verbose=1),
        TensorBoard(log_dir=f"logs/{model_name}/{datetime.datetime.now().strftime('%Y%m%d-%H%M%S')}",
                    histogram_freq=1)
    ]

    # Обучение модели
    history = model.fit(
        X_train, y_train,
        epochs=config['epochs'],
        batch_size=config['batch_size'],
        validation_data=(X_val, y_val),
        callbacks=callbacks,
        verbose=1
    )

    return history
