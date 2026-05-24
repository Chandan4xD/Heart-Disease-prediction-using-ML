import logging
import pickle
import warnings
from pathlib import Path
from typing import Tuple

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.svm import SVC
from sklearn.neighbors import KNeighborsClassifier
from sklearn.naive_bayes import GaussianNB
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score

warnings.filterwarnings('ignore')

logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s',
    datefmt='%Y-%m-%d %H:%M:%S'
)
logger = logging.getLogger(__name__)

class HeartDiseasePipeline:
    def __init__(self, data_path: str = 'heart.csv'):
        self.data_path = Path(data_path)
        self.models = {
            'Random Forest': RandomForestClassifier(n_estimators=100, random_state=42),
            'SVM': SVC(kernel='rbf', probability=True, random_state=42),
            'Decision Tree': DecisionTreeClassifier(random_state=42),
            'KNN': KNeighborsClassifier(n_neighbors=5),
            'Naive Bayes': GaussianNB()
        }
        self.trained_models = {}
        self.results = {}
        self.scaler = StandardScaler()
        self.feature_names = []
        self.best_model_name = ""

    def load_and_clean_data(self) -> pd.DataFrame:
        if not self.data_path.exists():
            raise FileNotFoundError(f"Dataset missing: {self.data_path.absolute()}")

        df = pd.read_csv(self.data_path)
        df.columns = df.columns.str.strip().str.replace('\ufeff', '')

        if 'patientid' in df.columns:
            df.drop('patientid', axis=1, inplace=True)
            
        if 'Classification' in df.columns and 'target' not in df.columns:
            df.rename(columns={'Classification': 'target'}, inplace=True)

        return df

    def generate_eda(self, df: pd.DataFrame, output_file: str = 'eda_plots.png') -> None:
        try:
            fig, axes = plt.subplots(2, 2, figsize=(12, 8))
            
            axes[0, 0].hist(df['age'], bins=20, color='steelblue', edgecolor='black')
            axes[0, 0].set_title('Age Distribution')
            
            df['target'].value_counts().plot(kind='bar', ax=axes[0, 1], color=['#2ecc71', '#e74c3c'])
            axes[0, 1].set_title('Target Distribution')
            axes[0, 1].tick_params(axis='x', rotation=0)

            sns.heatmap(df.corr()[['target']].sort_values('target', ascending=False), 
                        annot=True, fmt='.2f', cmap='Blues', ax=axes[1, 1])

            plt.tight_layout()
            plt.savefig(output_file, dpi=120)
            plt.close()
        except Exception as e:
            logger.error(f"EDA generation failed: {e}")

    def prepare_data(self, df: pd.DataFrame) -> Tuple[np.ndarray, np.ndarray, pd.Series, pd.Series]:
        X = df.drop('target', axis=1)
        y = df['target']
        
        self.feature_names = X.columns.tolist()
        
        X_train, X_test, y_train, y_test = train_test_split(
            X, y, test_size=0.2, random_state=42, stratify=y
        )

        X_train_scaled = self.scaler.fit_transform(X_train)
        X_test_scaled = self.scaler.transform(X_test)

        return X_train_scaled, X_test_scaled, y_train, y_test

    def train_and_evaluate(self, X_train: np.ndarray, X_test: np.ndarray, y_train: pd.Series, y_test: pd.Series) -> None:
        for name, model in self.models.items():
            model.fit(X_train, y_train)
            preds = model.predict(X_test)
            
            self.results[name] = {
                'Accuracy': round(accuracy_score(y_test, preds) * 100, 2),
                'Precision': round(precision_score(y_test, preds) * 100, 2),
                'Recall': round(recall_score(y_test, preds) * 100, 2),
                'F1-Score': round(f1_score(y_test, preds) * 100, 2)
            }
            self.trained_models[name] = model
            logger.info(f"{name} Acc: {self.results[name]['Accuracy']}%")

        self.best_model_name = max(self.results, key=lambda k: (self.results[k]['Accuracy'], self.results[k]['F1-Score']))
        logger.info(f"Selected model: {self.best_model_name}")

    def save_artifacts(self) -> None:
        artifacts = {
            'best_model.pkl': self.trained_models[self.best_model_name],
            'scaler.pkl': self.scaler,
            'results.pkl': self.results,
            'feature_names.pkl': self.feature_names
        }

        for filename, obj in artifacts.items():
            with open(filename, 'wb') as f:
                pickle.dump(obj, f)

    def execute(self) -> None:
        try:
            df = self.load_and_clean_data()
            self.generate_eda(df)
            X_train, X_test, y_train, y_test = self.prepare_data(df)
            self.train_and_evaluate(X_train, X_test, y_train, y_test)
            self.save_artifacts()
            logger.info("Pipeline finished.")
        except Exception as e:
            logger.exception("Pipeline aborted.")
            raise

if __name__ == "__main__":
    pipeline = HeartDiseasePipeline()
    pipeline.execute()
