

import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split, cross_val_score
from sklearn.preprocessing import StandardScaler
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier, VotingClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.svm import SVC
from sklearn.metrics import accuracy_score
from xgboost import XGBClassifier
from lightgbm import LGBMClassifier
from catboost import CatBoostClassifier
from mlxtend.classifier import StackingClassifier
import warnings
warnings.filterwarnings('ignore')

from sklearn.datasets import load_breast_cancer
data = load_breast_cancer()
X, y = data.data, data.target

scaler = StandardScaler()
X = scaler.fit_transform(X)
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

clf1 = LogisticRegression()
clf2 = RandomForestClassifier(n_estimators=100)
clf3 = XGBClassifier(use_label_encoder=False, eval_metric='logloss')
clf4 = LGBMClassifier()
clf5 = CatBoostClassifier(verbose=0)

voting_clf = VotingClassifier(
    estimators=[('lr', clf1), ('rf', clf2), ('xgb', clf3)],
    voting='soft'
)
voting_clf.fit(X_train, y_train)
y_pred = voting_clf.predict(X_test)
print("Voting Classifier Accuracy:", accuracy_score(y_test, y_pred))

stack_clf = StackingClassifier(
    classifiers=[clf1, clf2, clf3],
    meta_classifier=LogisticRegression()
)
stack_clf.fit(X_train, y_train)
y_pred_stack = stack_clf.predict(X_test)
print("Stacking Classifier Accuracy:", accuracy_score(y_test, y_pred_stack))

models = {
    "Logistic Regression": clf1,
    "Random Forest": clf2,
    "XGBoost": clf3,
    "LightGBM": clf4,
    "CatBoost": clf5,
    "Voting Ensemble": voting_clf,
    "Stacking Ensemble": stack_clf
}

for name, model in models.items():
    model.fit(X_train, y_train)
    y_pred = model.predict(X_test)
    print(f"{name} Accuracy: {accuracy_score(y_test, y_pred):.4f}")


import joblib
joblib.dump(voting_clf, 'voting_model.pkl')

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
 
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier, VotingClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix

from sklearn.datasets import load_iris
 
# Load dataset
data = load_iris()
X = pd.DataFrame(data.data, columns=data.feature_names)
y = pd.Series(data.target)
 
# Train-Test Split
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Define base models
model1 = LogisticRegression(max_iter=200)
model2 = RandomForestClassifier(n_estimators=100)
model3 = GradientBoostingClassifier()
 
# Ensemble using Voting
ensemble = VotingClassifier(estimators=[
    ('lr', model1), 
    ('rf', model2), 
    ('gb', model3)
], voting='hard')
 
# Fit models
model1.fit(X_train, y_train)
model2.fit(X_train, y_train)
model3.fit(X_train, y_train)
ensemble.fit(X_train, y_train)

models = {
    "Logistic Regression": model1,
    "Random Forest": model2,
    "Gradient Boosting": model3,
    "Ensemble (Voting)": ensemble
}
 
for name, model in models.items():
    preds = model.predict(X_test)
    print(f"--- {name} ---")
    print("Accuracy:", accuracy_score(y_test, preds))
    print("Classification Report:\n", classification_report(y_test, preds))




import streamlit as st
import pandas as pd
from sklearn.datasets import load_iris
from sklearn.ensemble import VotingClassifier, RandomForestClassifier, GradientBoostingClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score
 
st.set_page_config(page_title="Model Ensemble Dashboard", layout="centered")
st.title("🔮 Model Ensemble Dashboard")
 
data = load_iris()
X = pd.DataFrame(data.data, columns=data.feature_names)
y = pd.Series(data.target)
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
 
lr = LogisticRegression(max_iter=200)
rf = RandomForestClassifier(n_estimators=100)
gb = GradientBoostingClassifier()
 
ensemble = VotingClassifier(estimators=[
    ('lr', lr),
    ('rf', rf),
    ('gb', gb)
], voting='hard')
 
lr.fit(X_train, y_train)
rf.fit(X_train, y_train)
gb.fit(X_train, y_train)
ensemble.fit(X_train, y_train)
 
def evaluate_model(name, model):
    preds = model.predict(X_test)
    acc = accuracy_score(y_test, preds)
    st.subheader(name)
    st.write(f"Accuracy: **{acc:.2f}**")
 
evaluate_model("Logistic Regression", lr)
evaluate_model("Random Forest", rf)
evaluate_model("Gradient Boosting", gb)
evaluate_model("Voting Ensemble", ensemble)

# streamlit run ensemble_dashboard.py