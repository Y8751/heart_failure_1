import streamlit as st
import pandas as pd
import joblib
import seaborn as sns
import matplotlib.pyplot as plt
import numpy as np
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix
from tensorflow.keras.models import load_model

# ==============================
# Load Preprocessor
# ==============================
try:
    preprocessor = joblib.load("preprocessor_fixed.pkl")
except Exception as e:
    st.error(f"❌ Error loading preprocessor: {e}")
    st.stop()

# ==============================
# App Title
# ==============================
st.title("❤️ Heart Disease Classification App Using Multiple Models")

# ==============================
# Upload CSV File
# ==============================
uploaded_file = st.file_uploader("📂 Upload your CSV file", type=['csv'])

# ==============================
# Optional Sample CSV Download
# ==============================
try:
    with open("heart.csv", "rb") as f:
        st.download_button("📥 Download Sample CSV", f, file_name="heart.csv")
except FileNotFoundError:
    st.info("ℹ️ Sample CSV (heart.csv) not found")

# ==============================
# Load Data
# ==============================
@st.cache_resource
def load_data(file):
    return pd.read_csv(file)

# ==============================
# Preprocess Data
# ==============================
def preprocess_data(df):
    if 'HeartDisease' not in df.columns:
        st.error("❌ The uploaded file does not contain a 'HeartDisease' column.")
        st.stop()
    X = df.drop('HeartDisease', axis=1)
    y = df['HeartDisease']
    return X, y

# ==============================
# Load Models
# ==============================
@st.cache_resource
def load_models():
    models = {}
    try:
        models['XGBoost'] = joblib.load("XGBoost_model.pkl")
    except Exception as e:
        st.error(f"❌ Failed to load XGBoost model: {e}")
    try:
        models['SVM'] = joblib.load("SVM_model.pkl")
    except Exception as e:
        st.error(f"❌ Failed to load SVM model: {e}")
    try:
        models['Random Forest'] = joblib.load("Random_Forest_model.pkl")
    except Exception as e:
        st.error(f"❌ Failed to load Random Forest model: {e}")
    try:
        models['Keras Neural Network'] = load_model("keras_model.h5")
    except Exception as e:
        st.error(f"❌ Failed to load Keras model: {e}")
    if not models:
        st.stop()
    return models

# ==============================
# Prediction Function
# ==============================
def make_prediction(model, input_data, preprocessor):
    input_processed = preprocessor.transform(input_data)
    if 'keras' in str(type(model)).lower():
        prediction = model.predict(input_processed)
        return (prediction.flatten() > 0.5).astype(int)
    else:
        return model.predict(input_processed)

# ==============================
# Main App Logic
# ==============================
if uploaded_file is not None:
    df = load_data(uploaded_file)
    st.write("### 👁️ Preview of Uploaded Data", df.head())

    X, y = preprocess_data(df)
    models = load_models()

    # Select Model
    model_name = st.selectbox("🧠 Choose a Model", list(models.keys()))
    model = models[model_name]

    # Manual Input for Prediction
    st.subheader("✍️ Enter Patient Data for Prediction")
    input_data = {}
    for col in X.columns:
        if pd.api.types.is_numeric_dtype(X[col]):
            input_data[col] = st.number_input(f"{col}", value=float(X[col].mean()))
        else:
            input_data[col] = st.text_input(f"{col}", value=str(X[col].mode()[0]))

    if st.button("🔮 Predict Heart Disease"):
        try:
            input_df = pd.DataFrame([input_data])
            pred = make_prediction(model, input_df, preprocessor)
            st.success(f"✅ Prediction: {'Heart Disease' if pred[0] == 1 else 'No Heart Disease'}")
        except Exception as e:
            st.error(f"❌ Prediction failed: {e}")

    # Model Evaluation
    st.subheader("📊 Model Evaluation on Entire Dataset")
    try:
        X_processed = preprocessor.transform(X)
        if 'keras' in model_name.lower():
            y_pred = (model.predict(X_processed).flatten() > 0.5).astype(int)
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
    except Exception as e:
        st.error(f"❌ Error during evaluation: {e}")
else:
    st.info("👈 Please upload a CSV file to get started.")
