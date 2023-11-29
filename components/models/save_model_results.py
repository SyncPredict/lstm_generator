import json

import pandas as pd


def save_model_results(model, history, model_name):
    """
    Сохраняет результаты модели, включая веса и историю обучения.

    :param model: Обученная модель.
    :param history: История обучения модели.
    :param model_name: Имя модели для сохранения файлов.
    """
    # Сохранение весов модели
    model.save(f"models/{model_name}.h5")
    # Сохранение истории обучения
    pd.DataFrame(history.history).to_csv(f"models/{model_name}_history.csv")
