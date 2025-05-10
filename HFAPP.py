import streamlit as st
import pandas as pd
import joblib
import seaborn as sns
import matplotlib.pyplot as plt
import numpy as np
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix
from tensorflow.keras.models import load_model
import requests

# ==============================
# دالة لتحميل الملفات من GitHub
# ==============================
def download_file(url, filename):
    try:
        response = requests.get(url)
        response.raise_for_status()  # Raise an exception for HTTP errors
        with open(filename, 'wb') as f:
            f.write(response.content)
        print(f"تم تحميل الملف: {filename}")
    except Exception as e:
        st.error(f"حدث خطأ أثناء تحميل الملف {filename}: {e}")
        st.stop()

# رابط raw لتحميل الملفات من GitHub
preprocessor_url = "https://github.com/Y8751/heart_failure_1/raw/main/preprocessor.pkl"
xgboost_model_url = "https://github.com/Y8751/heart_failure_1/raw/main/XGBoost_model.pkl"
svm_model_url = "https://github.com/Y8751/heart_failure_1/raw/main/SVM_model.pkl"
random_forest_model_url = "https://github.com/Y8751/heart_failure_1/raw/main/Random_Forest_model.pkl"
keras_model_url = "https://github.com/Y8751/heart_failure_1/raw/main/keras_model.h5"

# تحميل الملفات من GitHub
download_file(preprocessor_url, "preprocessor.pkl")
download_file(xgboost_model_url, "XGBoost_model.pkl")
download_file(svm_model_url, "SVM_model.pkl")
download_file(random_forest_model_url, "Random_Forest_model.pkl")
download_file(keras_model_url, "keras_model.h5")

# ==============================
# Load Preprocessor with Error Handling
# ==============================
try:
    preprocessor = joblib.load("preprocessor.pkl")
except FileNotFoundError:
    st.error("Preprocessor file not found. Please ensure 'preprocessor.pkl' exists in the directory.")
    st.stop()
except Exception as e:
    st.error(f"Error loading preprocessor: {e}")
    st.stop()

# ==============================
# App Title
# ==============================
st.title("Heart Disease Classification App Using Multiple Models")

# ==============================
# Upload CSV File
# ==============================
uploaded_file = st.file_uploader("📂 Upload your CSV file", type=['csv'])

# ==============================
# Load Data
# ==============================
@st.cache_data
def load_data(file):
    return pd.read_csv(file)

# ==============================
# Preprocess Data
# ==============================
@st.cache_data
def preprocess_data(df):
    if 'HeartDisease' not in df.columns:
        st.error("The file does not contain the 'HeartDisease' column.")
        st.stop()
    X = df.drop('HeartDisease', axis=1)
    y = df['HeartDisease']
    return X, y

# ==============================
# Load Models with Error Handling
# ==============================
def load_models():
    try:
        models = {
            'XGBoost': joblib.load("XGBoost_model.pkl"),
            'SVM': joblib.load("SVM_model.pkl"),
            'Random Forest': joblib.load("Random_Forest_model.pkl"),
            'Keras Neural Network': load_model("keras_model.h5"),
        }
    except FileNotFoundError as e:
        st.error(f"Model file not found: {e.filename}")
        st.stop()
    except Exception as e:
        st.error(f"Error loading models: {e}")
        st.stop()
    return models

# ==============================
# Prediction Function
# ==============================
def make_prediction(model, input_data, preprocessor):
    input_processed = preprocessor.transform(input_data)
    if 'keras' in str(type(model)).lower():
        prediction = model.predict(input_processed)
        return (prediction > 0.5).astype(int)
    else:
        return model.predict(input_processed)

# ==============================
# Main App Logic
# ==============================
if uploaded_file is not None:
    df = load_data(uploaded_file)
    st.write("### 👁️‍🗨️ Data Preview", df.head())

    X, y = preprocess_data(df)
    models = load_models()

    # Select Model
    model_name = st.selectbox("Select a Model", list(models.keys()))
    model = models[model_name]

    # Manual Input for Prediction
    st.subheader("🔍 Enter Input Values for Prediction")
    input_data = {}
    for col in X.columns:
        input_data[col] = st.number_input(f"{col}", value=float(X[col].mean()))
    
    if st.button("Predict Heart Disease"):
        input_df = pd.DataFrame([input_data])
        pred = make_prediction(model, input_df, preprocessor)
        st.success(f"✅ Prediction: {'Heart Disease' if pred[0] == 1 else 'No Heart Disease'}")

    # Model Evaluation
    st.subheader("📊 Model Evaluation on Full Dataset")
    X_processed = preprocessor.transform(X)
    if 'keras' in model_name.lower():
        y_pred = (model.predict(X_processed) > 0.5).astype(int)
    else:
        y_pred = model.predict(X_processed)

    acc = accuracy_score(y, y_pred)
    st.success(f"🎯 Model Accuracy: {acc:.4f}")
    st.text("📄 Classification Report:")
    st.text(classification_report(y, y_pred))

    # Confusion Matrix
    cm = confusion_matrix(y, y_pred)
    fig, ax = plt.subplots()
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', ax=ax)
    ax.set_title("Confusion Matrix")
    ax.set_xlabel("Predicted")
    ax.set_ylabel("Actual")
    st.pyplot(fig)

else:
    st.info("👈 Please upload a CSV file containing the data.")
