from airflow import DAG
from airflow.operators.python_operator import PythonOperator
from airflow.utils.dates import days_ago
from datetime import timedelta
import mlflow
import logging
import sys
import os

# Добавляем путь к скриптам
sys.path.append('/opt/airflow/scripts')

from data_loader import download_titanic_data, save_data_locally, load_data_from_local
from data_preprocessor import preprocess_titanic_data

# Настройка MLflow
mlflow.set_tracking_uri("http://mlflow:5000")
mlflow.set_experiment("titanic_basic_pipeline")

default_args = {
    'owner': 'airflow',
    'depends_on_past': False,
    'start_date': days_ago(1),
    'retries': 1,
    'retry_delay': timedelta(minutes=5),
}

def download_and_log_data():
    """Задача 1: Загрузка данных и логирование в MLflow"""
    with mlflow.start_run(run_name="data_download"):
        # Загружаем данные
        train_df, test_df = download_titanic_data()
        
        # Логируем параметры и метрики
        mlflow.log_param("dataset", "titanic")
        mlflow.log_param("source", "kaggle")
        mlflow.log_metric("train_samples", len(train_df))
        mlflow.log_metric("test_samples", len(test_df))
        mlflow.log_metric("total_features", train_df.shape[1])
        
        # Логируем информацию о данных
        mlflow.log_text(str(train_df.info()), "data_info.txt")
        
        # Сохраняем данные для следующей задачи
        save_data_locally(train_df, test_df)
        
        logging.info("Данные успешно загружены и залогированы в MLflow")

def preprocess_and_log_data():
    """Задача 2: Предобработка данных и логирование"""
    with mlflow.start_run(run_name="data_preprocessing"):
        # Загружаем данные из предыдущей задачи
        train_df, test_df = load_data_from_local()
        
        # Предобработка
        processed_train, processed_test, encoders = preprocess_titanic_data(train_df, test_df)
        
        # Логируем параметры обработки
        mlflow.log_param("missing_value_strategy", "median/mode")
        mlflow.log_param("categorical_encoding", "label_encoding")
        
        # Логируем метрики после обработки
        mlflow.log_metric("processed_train_samples", len(processed_train))
        mlflow.log_metric("processed_test_samples", len(processed_test))
        mlflow.log_metric("processed_features", processed_train.shape[1])
        
        # Сохраняем обработанные данные
        save_data_locally(processed_train, processed_test, "/tmp/titanic_processed_final")
        
        # Логируем sample обработанных данных
        sample_data = processed_train.head(10).to_string()
        mlflow.log_text(sample_data, "processed_data_sample.txt")
        
        logging.info("Данные успешно обработаны и залогированы")

def log_dataset_summary():
    """Задача 3: Финальное логирование сводки"""
    with mlflow.start_run(run_name="dataset_summary"):
        train_df, test_df = load_data_from_local("/tmp/titanic_processed_final")
        
        # Создаем сводку
        summary = {
            "total_samples": len(train_df) + len(test_df),
            "train_samples": len(train_df),
            "test_samples": len(test_df),
            "features_count": train_df.shape[1],
            "memory_usage_mb": train_df.memory_usage(deep=True).sum() / 1024**2
        }
        
        # Логируем сводку
        for key, value in summary.items():
            mlflow.log_metric(key, value)
        
        # Логируем список колонок
        mlflow.log_param("features_list", str(list(train_df.columns)))
        
        logging.info(f"Сводка залогирована: {summary}")

# Определяем DAG
with DAG(
    'titanic_basic_pipeline',
    default_args=default_args,
    description='Базовый пайплайн для датасета Titanic',
    schedule_interval=None,  # Запуск только вручную
    catchup=False,
    tags=['titanic', 'mlflow', 'kaggle'],
) as dag:

    download_task = PythonOperator(
        task_id='download_titanic_data',
        python_callable=download_and_log_data,
    )

    preprocess_task = PythonOperator(
        task_id='preprocess_titanic_data',
        python_callable=preprocess_and_log_data,
    )

    summary_task = PythonOperator(
        task_id='log_dataset_summary',
        python_callable=log_dataset_summary,
    )

    download_task >> preprocess_task >> summary_task