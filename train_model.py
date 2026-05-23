import os
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
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score,
confusion_matrix
warnings.filterwarnings('ignore')
print("Starting model training pipeline...")
if not os.path.exists('heart.csv'):
print("Error: 'heart.csv' not found in the current directory.")
exit()
df = pd.read_csv('heart.csv')
print("Loaded 'heart.csv' successfully.")
# Cleaning up column names
df.columns = df.columns.str.strip()

# Dropping patientid as it's not a predictive feature
if 'patientid' in df.columns:
df.drop('patientid', axis=1, inplace=True)
# Ensuring target column is named correctly
if 'Classification' in df.columns and 'target' not in df.columns:
df.rename(columns={'Classification': 'target'}, inplace=True)
print("Generating EDA visualizations...")
fig, axes = plt.subplots(2, 2, figsize=(12, 8))
axes[0, 0].hist(df['age'], bins=20, color='steelblue', edgecolor='black')
axes[0, 0].set_title('Age Distribution')
df['target'].value_counts().plot(kind='bar', ax=axes[0, 1], color=['#2ecc71', '#e74c3c'])
axes[0, 1].set_title('Target Distribution (0=Healthy, 1=Disease)')
axes[0, 1].tick_params(axis='x', rotation=0)
sns.heatmap(df.corr()[['target']].sort_values('target', ascending=False),
annot=True, fmt='.2f', cmap='Blues', ax=axes[1, 1])
plt.tight_layout()
plt.savefig('eda_plots.png', dpi=120)
plt.close()
X = df.drop('target', axis=1)
# Saving feature names for consistency in Streamlit app
feature_names = X.columns.tolist()
y = df['target']
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42,
stratify=y)
scaler = StandardScaler()
X_train_s = scaler.fit_transform(X_train)
X_test_s = scaler.transform(X_test)
models = {
'Decision Tree': DecisionTreeClassifier(random_state=42),
'Random Forest': RandomForestClassifier(n_estimators=100, random_state=42),
'SVM': SVC(kernel='rbf', probability=True, random_state=42),
'KNN': KNeighborsClassifier(n_neighbors=5),
'Logistic Regression': LogisticRegression(max_iter=1000, random_state=42)
}
results = {}
trained_models = {}
for name, model in models.items():
model.fit(X_train_s, y_train)
preds = model.predict(X_test_s)
results[name] = {
'Accuracy': round(accuracy_score(y_test, preds) * 100, 2),
'Precision': round(precision_score(y_test, preds) * 100, 2),
'Recall': round(recall_score(y_test, preds) * 100, 2),

'F1-Score': round(f1_score(y_test, preds) * 100, 2)
}
trained_models[name] = model
print(f" -> {name}: {results[name]['Accuracy']}%")
best_algo = max(results, key=lambda k: results[k]['Accuracy'])
print(f"\nBest Model: {best_algo}")
with open('best_model.pkl', 'wb') as f: pickle.dump(trained_models[best_algo], f)
with open('scaler.pkl', 'wb') as f: pickle.dump(scaler, f)
with open('results.pkl', 'wb') as f: pickle.dump(results, f)
with open('feature_names.pkl', 'wb') as f: pickle.dump(feature_names, f)
print("Done! Artifacts saved.")
