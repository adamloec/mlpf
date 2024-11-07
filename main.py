import sys
import pandas as pd
import numpy as np
from sklearn.datasets import load_iris
from sklearn.model_selection import train_test_split

from mlflow import log_metric, log_params, start_run

from pipeline import MLPipeline

def mlpf(model_name):
    iris = load_iris()
    df = pd.DataFrame(iris.data, columns=iris.feature_names)
    df['target'] = iris.target
    
    X = df.drop('target', axis=1)
    y = df['target']
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
    
    pipeline = MLPipeline()
    if model_name not in pipeline.models:
        print(f"MLPF: {model_name} not supported or does not exist.")
    
    numeric_cols = iris.feature_names
    X_train_processed = pipeline.prepare_features(X_train, numeric_columns=numeric_cols)
    X_test_processed = pipeline.prepare_features(X_test, numeric_columns=numeric_cols)
    
    model = pipeline.train(model_name, X_train_processed, y_train)
    train_score = model.score(X_train_processed, y_train)
    test_score = model.score(X_test_processed, y_test)

    return train_score, test_score

if __name__ == "__main__":
    print("MLPF: Starting Machine Learning Pipeline Framework...")
    print("""
    MLPF: Supported models
          'logistic' = LogisticRegression()
          'decision_tree' = DecisionTreeClassifier()
          'random_forest' = RandomForestClassifier()
    """)

    if len(sys.argv) < 1:
        print("Usage: python main.py MODEL_NAME")
        sys.exit(1)

    model_name = sys.argv[1]
    train_score, test_score = mlpf(model_name)

    print("-"*50)
    print(f"Results for {model_name}")
    print(f"MLPF: Train Score: {train_score}")
    print(f"MLPF: {test_score}")
    print("-"*50)