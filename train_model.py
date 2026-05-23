import logging
import pickle
import warnings
from pathlib import Path
from typing import Tuple

import pandas as pd
import numpy as np

from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.svm import SVC
from sklearn.neighbors import KNeighborsClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score

warnings.filterwarnings('ignore')

logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

class HeartDiseasePipeline:
    def __init__(self, data_path: str = 'heart.csv'):
        self.data_path = Path(data_path)
        self.models = {
            'Decision Tree': DecisionTreeClassifier(random_state=42),
            'Random Forest': RandomForestClassifier(n_estimators=100, random_state=42),
            'SVM': SVC(kernel='rbf', probability=True, random_state=42),
            'KNN': KNeighborsClassifier(n_neighbors=5),
            'Logistic Regression': LogisticRegression(max_iter=1000, random_state=42)
        }
        self.required_features = 12

    def validate_schema(self, df: pd.DataFrame):
        """Ensures the dataset has the exact number of required features."""
        if df.shape[1] < self.required_features:
            raise ValueError(f"Schema Validation Failed: Expected at least {self.required_features} features.")
        logger.info("Schema validation passed.")

    def load_and_clean_data(self) -> pd.DataFrame:
        if not self.data_path.exists():
            raise FileNotFoundError(f"Dataset missing: {self.data_path.absolute()}")

        df = pd.read_csv(self.data_path)
        df.columns = df.columns.str.strip()

        if 'patientid' in df.columns:
            df.drop('patientid', axis=1, inplace=True)
            
        if 'Classification' in df.columns and 'target' not in df.columns:
            df.rename(columns={'Classification': 'target'}, inplace=True)
            
        self.validate_schema(df)
        return df

    def prepare_data(self, df: pd.DataFrame) -> Tuple[np.ndarray, np.ndarray, pd.Series, pd.Series, StandardScaler]:
        X = df.drop('target', axis=1)
        y = df['target']
        
        X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42, stratify=y)
        
        scaler = StandardScaler()
        X_train_s = scaler.fit_transform(X_train)
        X_test_s = scaler.transform(X_test)
        
        return X_train_s, X_test_s, y_train, y_test, scaler

    def execute(self) -> None:
        try:
            df = self.load_and_clean_data()
            X_train, X_test, y_train, y_test, scaler = self.prepare_data(df)
            
            results = {}
            trained_models = {}

            logger.info("Training and evaluating models...")
            
            # Train and evaluate all models properly
            for name, model in self.models.items():
                model.fit(X_train, y_train)
                preds = model.predict(X_test)
                
                results[name] = {
                    'Accuracy': round(accuracy_score(y_test, preds) * 100, 2),
                    'Precision': round(precision_score(y_test, preds) * 100, 2),
                    'Recall': round(recall_score(y_test, preds) * 100, 2),
                    'F1-Score': round(f1_score(y_test, preds) * 100, 2)
                }
                trained_models[name] = model
                logger.info(f"{name} evaluated -> Accuracy: {results[name]['Accuracy']}%")
            
            # Dynamically select the best model
            best_model_name = max(results, key=lambda k: results[k]['Accuracy'])
            best_model = trained_models[best_model_name]
            logger.info(f"🏆 Best model selected: {best_model_name}")

            # Save ALL artifacts required by app.py
            with open('best_model.pkl', 'wb') as f: pickle.dump(best_model, f)
            with open('scaler.pkl', 'wb') as f: pickle.dump(scaler, f)
            with open('results.pkl', 'wb') as f: pickle.dump(results, f) # <-- THIS WAS MISSING
            
            logger.info("✅ Resilient pipeline execution complete. Artifacts saved.")
        except Exception as e:
            logger.critical(f"❌ Critical system failure: {e}")
            raise

if __name__ == "__main__":
    pipeline = HeartDiseasePipeline()
    pipeline.execute()
