import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import train_test_split, RandomizedSearchCV
from sklearn.preprocessing import StandardScaler
from sklearn.svm import SVC
from sklearn.metrics import accuracy_score, confusion_matrix, classification_report, roc_auc_score, roc_curve
import joblib
from imblearn.over_sampling import SMOTE

url = 'heart+disease/processed.cleveland.data'
columns = ['age', 'sex', 'cp', 'trestbps', 'chol', 'fbs', 'restecg', 'thalach', 'exang', 'oldpeak', 'slope', 'ca', 'thal', 'target']
data = pd.read_csv(url, header=None, names=columns)

data.replace('?', np.nan, inplace=True)

data['ca'] = pd.to_numeric(data['ca'])
data['thal'] = pd.to_numeric(data['thal'])

data['ca'].fillna(data['ca'].median(), inplace=True)
data['thal'].fillna(data['thal'].median(), inplace=True)

data['target'] = data['target'].apply(lambda x: 1 if x > 0 else 0)

print(data['target'].value_counts())

X = data.drop('target', axis=1)
y = data['target']

smote = SMOTE(random_state=42)
X_resampled, y_resampled = smote.fit_resample(X, y)

X_train, X_test, y_train, y_test = train_test_split(X_resampled, y_resampled, test_size=0.2, random_state=42)

scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled = scaler.fit_transform(X_test)

svm_model = SVC(probability=True, random_state=42, class_weight='balanced')

param_dist = {
    'C': [0.1, 1, 10, 100],
    'gamma': ['scale', 'auto', 0.1, 1],
    'kernel': ['linear', 'rbf']
}

random_search = RandomizedSearchCV(svm_model, param_distributions=param_dist, n_iter=10, cv=3, scoring='accuracy', random_state=42)

random_search.fit(X_train_scaled, y_train)

print("\nBest Hyperparameters from Random Search:", random_search.best_params_)
print("Best Random Search Accuracy:", random_search.best_score_)

best_random_model = random_search.best_estimator_
y_pred_random = best_random_model.predict(X_test_scaled)

print("Test Accuracy (Random Search): ", accuracy_score(y_test, y_pred_random) * 100)

print("\nRandom Search Confusion Matrix:\n", confusion_matrix(y_test, y_pred_random))
print("\nRandom Search Classification Report:\n", classification_report(y_test, y_pred_random))

random_roc_auc = roc_auc_score(y_test, best_random_model.predict(X_test_scaled))
fpr_random, tpr_random, _ = roc_curve(y_test, best_random_model.predict_proba(X_test_scaled)[:, 1])

joblib.dump(best_random_model, 'heart_disease_svm_model.pkl')

joblib.dump(scaler, 'scaler.pkl')

plt.figure()
plt.plot(fpr_random, tpr_random, label='Random Search SVM (area = %0.2f)' % random_roc_auc)
plt.plot([0, 1], [0, 1], 'k--')
plt.xlim([0.0, 1.0])
plt.ylim([0.0, 1.05])
plt.xlabel('False Positive Rate')
plt.ylabel('True Positive Rate')
plt.title('ROC Curve - Random Search')
plt.legend(loc="lower right")
plt.show()