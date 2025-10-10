import pandas as pd
import os
from kaggle.api.kaggle_api_extended import KaggleApi
import logging

def download_titanic_data():
    """Загрузка датасета Titanic с Kaggle"""
    try:
        # Настройка Kaggle API
        api = KaggleApi()
        api.authenticate()
        
        # Скачивание датасета
        dataset_name = "heptapod/titanic"
        download_path = "/tmp/titanic_data"
        
        api.dataset_download_files(dataset_name, path=download_path, unzip=True)
        
        # Загрузка данных
        train_df = pd.read_csv(f"{download_path}/train.csv")
        test_df = pd.read_csv(f"{download_path}/test.csv")
        
        logging.info(f"Данные загружены. Train shape: {train_df.shape}, Test shape: {test_df.shape}")
        
        return train_df, test_df
        
    except Exception as e:
        logging.error(f"Ошибка при загрузке данных: {e}")
        raise

def save_data_locally(train_df, test_df, path="/tmp/titanic_processed"):
    """Сохранение данных для передачи между задачами"""
    os.makedirs(path, exist_ok=True)
    train_df.to_csv(f"{path}/train.csv", index=False)
    test_df.to_csv(f"{path}/test.csv", index=False)
    
def load_data_from_local(path="/tmp/titanic_processed"):
    """Загрузка данных из локального хранилища"""
    train_df = pd.read_csv(f"{path}/train.csv")
    test_df = pd.read_csv(f"{path}/test.csv")
    return train_df, test_df