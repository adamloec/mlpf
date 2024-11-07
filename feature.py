import pandas as pd
from sklearn.preprocessing import StandardScaler, MinMaxScaler

class FeatureEngineer:
    def __init__(self):
        self.transformers = {}
        self.numeric_col = []
        self.categorical_columns = []

    def add_numeric_features(self, columns):
        self.numeric_columns = columns
        self.transformers['standard_scaler'] = StandardScaler()

    def add_categorical_features(self, columns):
        self.categorical_columns = columns

    def create_features(self, data: pd.DataFrame) -> pd.DataFrame:
        df = data.copy()

        if self.numeric_columns:
            df[self.numeric_columns] = self.transformers['standard_scaler'].fit_transform(df[self.numeric_columns])

        return df