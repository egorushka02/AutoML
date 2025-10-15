import pandas as pd
import numpy as np
from sklearn.preprocessing import LabelEncoder
import logging

def validate_titanic_data(df, expected_columns=None):
    """Валидация данных Titanic"""
    if expected_columns is None:
        expected_columns = ['PassengerId', 'Pclass', 'Sex', 'Age', 'SibSp', 'Parch', 'Fare', 'Embarked']
    
    # Проверяем, что DataFrame не пустой
    if df.empty:
        logging.error("DataFrame пустой")
        return False
    
    # Проверяем наличие обязательных колонок
    missing_cols = [col for col in expected_columns if col not in df.columns]
    if missing_cols:
        logging.warning(f"Отсутствуют колонки: {missing_cols}")
        return False
        
    logging.info(f"Валидация прошла успешно. Размер данных: {df.shape}")
    return True

def preprocess_titanic_data(train_df, test_df):
    """Базовая предобработка данных Titanic"""
    
    # Валидируем входные данные
    if not validate_titanic_data(train_df):
        raise ValueError("Ошибка валидации тренировочных данных")
    
    if not validate_titanic_data(test_df, ['PassengerId', 'Pclass', 'Sex', 'Age', 'SibSp', 'Parch', 'Fare', 'Embarked']):
        raise ValueError("Ошибка валидации тестовых данных")
    
    # Объединяем для согласованной обработки
    combined = pd.concat([train_df, test_df], ignore_index=True)
    
    # Логируем исходные размеры
    logging.info(f"Исходные данные - Train: {train_df.shape}, Test: {test_df.shape}")
    
    # Обработка пропущенных значений с проверкой
    try:
        if 'Age' in combined.columns:
            age_median = combined['Age'].median()
            if pd.isna(age_median):
                age_median = 30  # Значение по умолчанию
            combined['Age'].fillna(age_median, inplace=True)
            logging.info(f"Заполнено {combined['Age'].isna().sum()} пропущенных значений в Age медианой: {age_median}")
        
        if 'Embarked' in combined.columns:
            embarked_mode = combined['Embarked'].mode()
            if len(embarked_mode) > 0:
                combined['Embarked'].fillna(embarked_mode[0], inplace=True)
            else:
                combined['Embarked'].fillna('S', inplace=True)  # Значение по умолчанию
            logging.info(f"Заполнено {combined['Embarked'].isna().sum()} пропущенных значений в Embarked")
        
        if 'Fare' in combined.columns:
            fare_median = combined['Fare'].median()
            if pd.isna(fare_median):
                fare_median = 15.0  # Значение по умолчанию
            combined['Fare'].fillna(fare_median, inplace=True)
            logging.info(f"Заполнено {combined['Fare'].isna().sum()} пропущенных значений в Fare медианой: {fare_median}")
            
    except Exception as e:
        logging.error(f"Ошибка при обработке пропущенных значений: {e}")
        raise
    
    # Создание новых признаков
    combined['FamilySize'] = combined['SibSp'] + combined['Parch'] + 1
    combined['IsAlone'] = (combined['FamilySize'] == 1).astype(int)
    
    # Кодирование категориальных переменных
    label_encoders = {}
    categorical_cols = ['Sex', 'Embarked']
    
    for col in categorical_cols:
        le = LabelEncoder()
        combined[col] = le.fit_transform(combined[col].astype(str))
        label_encoders[col] = le
    
    # Разделяем обратно
    processed_train = combined.iloc[:len(train_df)].copy()
    processed_test = combined.iloc[len(train_df):].copy()
    
    # Удаляем целевой признак из тестовых данных
    if 'Survived' in processed_test.columns:
        processed_test = processed_test.drop('Survived', axis=1)
    
    logging.info(f"После обработки - Train: {processed_train.shape}, Test: {processed_test.shape}")
    
    return processed_train, processed_test, label_encoders