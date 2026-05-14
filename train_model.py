"""
Heart Disease Prediction - Model Training Pipeline
Author: Chandan Kumar Mishra
Institution: B.A. College of Engineering and Technology (BACET)
"""

import pickle
import warnings
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
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, confusion_matrix

# Mute sklearn warnings to keep the console clean
warnings.filterwarnings('ignore')

def main():
    print("Starting the Heart Disease model training pipeline...\n")

    # 1. Load Data
    print("Fetching the dataset from UCI repository...")
    url = "https://archive.ics.uci.edu/ml/machine-learning-databases/heart-disease/processed.cleveland.data"
    cols = ['age', 'sex', 'cp', 'trestbps', 'chol', 'fbs', 'restecg', 
            'thalach', 'exang', 'oldpeak', 'slope', 'ca', 'thal', 'target']

    try:
        df = pd.read_csv(url, names=cols, na_values='?')
        print(f"Data loaded successfully: {df.shape[0]} rows, {df.shape[1]} columns.")
    except Exception as e:
        print(f"Error loading data: {e}")
        return

    # 2. Clean & Preprocess
    print("Cleaning missing values and formatting columns...")
    df.dropna(inplace=True)

    # Convert target to binary (0 = healthy, 1 = heart disease)
    df['target'] = df['target'].apply(lambda x: 1 if x > 0 else 0)

    # Cast specific columns to integers
    df['ca'] = df['ca'].astype(int)
    df['thal'] = df['thal'].astype(int)

    # 3. EDA & Plotting
    print("Generating EDA plots...")
    fig, axes = plt.subplots(2, 2, figsize=(14, 10))
    fig.suptitle("Heart Disease Dataset Analysis", fontsize=16)

    # Age histogram
    axes[0, 0].hist(df['age'], bins=20, color='steelblue', edgecolor='black')
    axes[0, 0].set_title('Age Distribution')

    # Target class balance
    df['target'].value_counts().plot(kind='bar', ax=axes[0, 1], color=['#2ecc71', '#e74c3c'])
    axes[0, 1].set_title('Disease Prevalence (0 = No, 1 = Yes)')
    axes[0, 1].tick_params(axis='x', rotation=0)

    # Scatter plot
    axes[1, 0].scatter(df['age'], df['chol'], c=df['target'], cmap='RdYlGn_r', alpha=0.6)
    axes[1, 0].set_title('Cholesterol vs. Age')

    # Correlation matrix
    sns.heatmap(df.corr()[['target']].sort_values('target', ascending=False),
                annot=True, fmt='.2f', cmap='coolwarm', ax=axes[1, 1])
    axes[1, 1].set_title('Feature Correlation')

    plt.tight_layout()
    plt.savefig('eda_plots.png', dpi=150)
    plt.close()

    # 4. Train/Test Split
    X = df.drop('target', axis=1)
    y = df['target']
    
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42, stratify=y)

    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train)
    X_test_scaled = scaler.transform(X_test)

    # 5. Model Training
    print("\nTraining classification models...")
    classifiers = {
        'Decision Tree': DecisionTreeClassifier(random_state=42),
        'Random Forest': RandomForestClassifier(n_estimators=100, random_state=42),
        'SVM': SVC(kernel='rbf', probability=True, random_state=42),
        'KNN': KNeighborsClassifier(n_neighbors=5),
        'Logistic Regression': LogisticRegression(max_iter=1000, random_state=42)
    }

    metrics = {}
    for name, clf in classifiers.items():
        clf.fit(X_train_scaled, y_train)
        preds = clf.predict(X_test_scaled)
        
        metrics[name] = {
            'Accuracy': accuracy_score(y_test, preds) * 100,
            'Precision': precision_score(y_test, preds) * 100,
            'Recall': recall_score(y_test, preds) * 100,
            'F1': f1_score(y_test, preds) * 100
        }
        print(f" -> {name} trained (Accuracy: {metrics[name]['Accuracy']:.1f}%)")

    # 6. Evaluate and grab the best model
    results_df = pd.DataFrame(metrics).T
    best_algo = results_df['Accuracy'].idxmax()
    best_model = classifiers[best_algo]

    print(f"\nTop performing model: {best_algo} with {results_df.loc[best_algo, 'Accuracy']:.1f}% accuracy")

    # Export confusion matrix for the winning model
    cm = confusion_matrix(y_test, best_model.predict(X_test_scaled))
    plt.figure(figsize=(5, 4))
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues')
    plt.title(f'Confusion Matrix: {best_algo}')
    plt.ylabel('Actual')
    plt.xlabel('Predicted')
    plt.savefig('confusion_matrix.png', dpi=150, bbox_inches='tight')
    plt.close()

    # 7. Save Artifacts
    print("\nExporting model artifacts...")
    with open('best_model.pkl', 'wb') as f:
        pickle.dump(best_model, f)
    with open('scaler.pkl', 'wb') as f:
        pickle.dump(scaler, f)
    with open('results.pkl', 'wb') as f:
        pickle.dump(metrics, f)

    print("Done! Files saved successfully. You can now launch Streamlit.")

if __name__ == '__main__':
    main()
