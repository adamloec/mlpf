import pandas as pd
import numpy as np
from sklearn.linear_model import LogisticRegression
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier

from feature import FeatureEngineer

MODELS = {
        'logistic': LogisticRegression(),
        'decision_tree': DecisionTreeClassifier(),
        'random_forest': RandomForestClassifier()
    }

class MLPipeline:
    def __init__(self):
        self.models = MODELS
        self.current_experiment = None
        self.results = {}
        self.feature_engineer = FeatureEngineer()
    
    def prepare_features(self, data: pd.DataFrame, numeric_columns=None, categorical_columns=None):
        if numeric_columns:
            self.feature_engineer.add_numeric_features(numeric_columns)
        if categorical_columns:
            self.feature_engineer.add_categorical_features(categorical_columns)
        
        return self.feature_engineer.create_features(data)

    def preprocess_data(self, data: pd.DataFrame) -> pd.DataFrame:
        data = data.dropna()

        for column in data.select_dtypes(['object']):
            data[column] = pd.Categorical(data[column]).codes

        return data

    def train(self, name, X_train, y_train):
        if name not in self.models:
            raise ValueError(f"Model {name} not found.")

        model = self.models[name]
        model.fit(X_train, y_train)
        return model