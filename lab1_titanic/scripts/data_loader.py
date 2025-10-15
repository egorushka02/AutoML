import pandas as pd
import os
import logging

def download_titanic_data():
    """Загрузка датасета Titanic с прямых ссылок"""
    try:
        # Используем прямые ссылки на датасет Titanic
        train_url = "https://raw.githubusercontent.com/datasciencedojo/datasets/master/titanic.csv"
        
        # Загружаем данные с обработкой ошибок
        try:
            train_df = pd.read_csv(train_url)
        except Exception as e:
            logging.error(f"Ошибка при загрузке данных с URL {train_url}: {e}")
            raise
        
        # Проверяем, что данные загружены
        if train_df.empty:
            raise ValueError("Загружен пустой датасет")
        
        # Проверяем наличие обязательных колонок
        required_columns = ['PassengerId', 'Survived', 'Pclass', 'Sex', 'Age', 'SibSp', 'Parch', 'Fare', 'Embarked']
        missing_columns = [col for col in required_columns if col not in train_df.columns]
        if missing_columns:
            logging.warning(f"Отсутствуют колонки: {missing_columns}")
            # Попробуем альтернативные названия
            if 'survived' in train_df.columns and 'Survived' not in train_df.columns:
                train_df = train_df.rename(columns={'survived': 'Survived'})
                logging.info("Переименована колонка 'survived' в 'Survived'")
        
        # Создаем тестовый набор из части тренировочного (для демонстрации)
        # В реальном проекте у вас будет отдельный test.csv
        test_size = int(len(train_df) * 0.2)
        test_df = train_df.tail(test_size).copy()
        train_df = train_df.head(len(train_df) - test_size).copy()
        
        # Удаляем целевую переменную из тестового набора
        if 'Survived' in test_df.columns:
            test_df = test_df.drop('Survived', axis=1)
        elif 'survived' in test_df.columns:
            test_df = test_df.drop('survived', axis=1)
        
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