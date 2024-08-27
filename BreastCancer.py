import streamlit as st
import lightgbm as lgb
import xgboost as xgb
from sklearn.ensemble import RandomForestClassifier
import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, classification_report, confusion_matrix
from sklearn.datasets import load_breast_cancer
import matplotlib.pyplot as plt
import seaborn as sns
import time

# Streamlit başlığı
st.title("Breast Cancer Prediction with LightGBM, Random Forest, and XGBoost")
st.write("Bu projede, LightGBM, Random Forest ve XGBoost modelleri ile göğüs kanseri sınıflandırması yapacağız.")

# Veri yükleme
data = load_breast_cancer(as_frame=True)
X = data.data
y = data.target

# Kullanıcıya veri çerçevesini görüntüleme seçeneği sunma
if st.checkbox('Show dataframe'):
    st.write(X)

# NumPy dizisini pandas DataFrame'e çevirme
X = pd.DataFrame(X)

# Verileri eğitim ve test setlerine ayırma
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Sayısal verileri ayırma
numerical_columns = X.select_dtypes(include=['float64', 'int64']).columns.tolist()

# Sayısal veri seçimi ve görselleştirme seçenekleri
st.sidebar.header("Görselleştirme Seçenekleri")
selected_column = st.sidebar.selectbox("Sayısal Kolon Seçin", numerical_columns)

plot_type = st.sidebar.selectbox("Grafik Tipi Seçin", ("Histogram", "Violin Plot", "Box Plot"))

if plot_type == "Histogram":
    fig, ax = plt.subplots()
    ax.hist(X[selected_column], bins=20)
    ax.set_title(f"Histogram of {selected_column}")
    st.pyplot(fig)

elif plot_type == "Violin Plot":
    fig, ax = plt.subplots()
    sns.violinplot(x=y, y=X[selected_column], ax=ax)
    ax.set_title(f"Violin Plot of {selected_column}")
    st.pyplot(fig)

elif plot_type == "Box Plot":
    fig, ax = plt.subplots()
    sns.boxplot(x=y, y=X[selected_column], ax=ax)
    ax.set_title(f"Box Plot of {selected_column}")
    st.pyplot(fig)

# LightGBM modeli eğitme ve değerlendirme
st.header("Model Training and Evaluation")

# LightGBM
train_data = lgb.Dataset(X_train, label=y_train)
params = {
    "objective": "binary",
    "boosting_type": "rf",
    "num_leaves": 5,
    "force_row_wise": True,
    "learning_rate": 0.5,
    "metric": "binary_logloss",
    "bagging_fraction": 0.8,
    "feature_fraction": 0.8
}

num_round = 500
start_time = time.time()
bst = lgb.train(params, train_data, num_round)
lgb_time = time.time() - start_time

y_pred_lgb = bst.predict(X_test)
y_pred_lgb_binary = (y_pred_lgb > 0.5).astype(int)

# Random Forest
rf_model = RandomForestClassifier(n_estimators=100, random_state=42)
start_time = time.time()
rf_model.fit(X_train, y_train)
rf_time = time.time() - start_time

y_pred_rf = rf_model.predict(X_test)

# XGBoost
xgb_model = xgb.XGBClassifier(learning_rate=0.1, n_estimators=100, use_label_encoder=False, eval_metric='logloss')
start_time = time.time()
xgb_model.fit(X_train, y_train)
xgb_time = time.time() - start_time

y_pred_xgb = xgb_model.predict(X_test)

# Model değerlendirme
def evaluate_model(y_test, y_pred, model_name, time_taken):
    accuracy = accuracy_score(y_test, y_pred)
    precision = precision_score(y_test, y_pred)
    recall = recall_score(y_test, y_pred)
    f1score = f1_score(y_test, y_pred)
    return {
        "Model": model_name,
        "Accuracy": accuracy,
        "Precision": precision,
        "Recall": recall,
        "F1-Score": f1score,
        "Time (s)": time_taken
    }

metrics_lgb = evaluate_model(y_test, y_pred_lgb_binary, "LightGBM", lgb_time)
metrics_rf = evaluate_model(y_test, y_pred_rf, "Random Forest", rf_time)
metrics_xgb = evaluate_model(y_test, y_pred_xgb, "XGBoost", xgb_time)

metrics_df = pd.DataFrame([metrics_lgb, metrics_rf, metrics_xgb])

# Sonuçları Streamlit'te gösterme
st.header("Model Performance Comparison")
st.dataframe(metrics_df)

# Classification report'u bir DataFrame'e dönüştürme ve gösterme
def show_classification_report(y_test, y_pred, model_name):
    st.subheader(f"{model_name} Classification Report")
    report = classification_report(y_test, y_pred, output_dict=True)
    report_df = pd.DataFrame(report).transpose()
    st.dataframe(report_df)

show_classification_report(y_test, y_pred_lgb_binary, "LightGBM")
show_classification_report(y_test, y_pred_rf, "Random Forest")
show_classification_report(y_test, y_pred_xgb, "XGBoost")

# Karışıklık matrisi gösterme
def plot_confusion_matrix(y_test, y_pred, model_name):
    st.subheader(f"{model_name} Confusion Matrix")
    cm = confusion_matrix(y_test, y_pred)
    fig, ax = plt.subplots()
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', ax=ax)
    ax.set_title(f"{model_name} Confusion Matrix")
    ax.set_xlabel("Predicted")
    ax.set_ylabel("True")
    st.pyplot(fig)

plot_confusion_matrix(y_test, y_pred_lgb_binary, "LightGBM")
plot_confusion_matrix(y_test, y_pred_rf, "Random Forest")
plot_confusion_matrix(y_test, y_pred_xgb, "XGBoost")
