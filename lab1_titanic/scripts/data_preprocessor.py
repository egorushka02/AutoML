import pandas as pd
import numpy as np
from sklearn.preprocessing import LabelEncoder
import logging

def preprocess_titanic_data(train_df, test_df):
    """Базовая предобработка данных Titanic"""
    
    # Объединяем для согласованной обработки
    combined = pd.concat([train_df, test_df], ignore_index=True)
    
    # Логируем исходные размеры
    logging.info(f"Исходные данные - Train: {train_df.shape}, Test: {test_df.shape}")
    
    # Обработка пропущенных значений
    combined['Age'].fillna(combined['Age'].median(), inplace=True)
    combined['Embarked'].fillna(combined['Embarked'].mode()[0], inplace=True)
    combined['Fare'].fillna(combined['Fare'].median(), inplace=True)
    
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