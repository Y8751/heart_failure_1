import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix
from sklearn.impute import SimpleImputer
from sklearn.linear_model import LogisticRegression, RidgeClassifier, Lasso
from sklearn.naive_bayes import GaussianNB
from sklearn.neighbors import KNeighborsClassifier
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.svm import SVC
from xgboost import XGBClassifier
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, Dropout

# Title
st.title("Heart Disease Classification App")

# Upload CSV
uploaded_file = st.file_uploader("Upload your CSV file", type=['csv'])

@st.cache_data
def load_data(file):
    df = pd.read_csv(file)
    return df

# Remove outliers using IQR
def remove_outliers_iqr(data, columns, multiplier=1.5):
    df_clean = data.copy()
    for col in columns:
        Q1 = df_clean[col].quantile(0.25)
        Q3 = df_clean[col].quantile(0.75)
        IQR = Q3 - Q1
        lower_bound = Q1 - multiplier * IQR
        upper_bound = Q3 + multiplier * IQR
        df_clean = df_clean[(df_clean[col] >= lower_bound) & (df_clean[col] <= upper_bound)]
    return df_clean

# Preprocessing
@st.cache_data
def preprocess_data(df):
    df = remove_outliers_iqr(df, df.select_dtypes(include=['int64', 'float64']).columns)
    X = df.drop('HeartDisease', axis=1)
    y = df['HeartDisease']
    
    numeric_features = X.select_dtypes(include=['int64', 'float64']).columns
    categorical_features = X.select_dtypes(include=['object']).columns

    numeric_transformer = Pipeline(steps=[
        ('imputer', SimpleImputer(strategy='median')),
        ('scaler', StandardScaler())])

    categorical_transformer = Pipeline(steps=[
        ('imputer', SimpleImputer(strategy='most_frequent')),
        ('encoder', OneHotEncoder(handle_unknown='ignore'))])

    preprocessor = ColumnTransformer([
        ('num', numeric_transformer, numeric_features),
        ('cat', categorical_transformer, categorical_features)
    ])

    return X, y, preprocessor, numeric_features, categorical_features

# Modeling functions
def get_models():
    return {
        'Logistic Regression': LogisticRegression(max_iter=1000),
        'KNN': KNeighborsClassifier(),
        'Decision Tree': DecisionTreeClassifier(),
        'Random Forest': RandomForestClassifier(),
        'SVM': SVC(probability=True),
        'Naive Bayes': GaussianNB(),
        'Ridge Classifier': RidgeClassifier(),
        'Lasso (Logistic Approx)': LogisticRegression(penalty='l1', solver='liblinear'),
        'XGBoost': XGBClassifier(use_label_encoder=False, eval_metric='logloss')
    }

# Keras model separately
def build_keras_model(input_shape):
    model = Sequential([
        Dense(64, activation='relu', input_shape=(input_shape,)),
        Dropout(0.3),
        Dense(32, activation='relu'),
        Dropout(0.3),
        Dense(1, activation='sigmoid')
    ])
    model.compile(optimizer='adam', loss='binary_crossentropy', metrics=['accuracy'])
    return model

if uploaded_file is not None:
    df = load_data(uploaded_file)
    st.write("### Dataset Preview", df.head())
    X, y, preprocessor, numeric_features, categorical_features = preprocess_data(df)
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

    model_name = st.selectbox("Select a model", list(get_models().keys()) + ['Neural Network (Keras)'])
    
    if model_name != 'Neural Network (Keras)':
        model = get_models()[model_name]
        clf = Pipeline([
            ('preprocessor', preprocessor),
            ('classifier', model)
        ])
        clf.fit(X_train, y_train)
        y_pred = clf.predict(X_test)
    
    else:
        X_train_processed = preprocessor.fit_transform(X_train)
        X_test_processed = preprocessor.transform(X_test)
        model = build_keras_model(X_train_processed.shape[1])
        model.fit(X_train_processed, y_train, epochs=50, validation_split=0.2, verbose=0)
        y_pred_prob = model.predict(X_test_processed)
        y_pred = (y_pred_prob > 0.5).astype("int32")

    acc = accuracy_score(y_test, y_pred)
    st.success(f"Model Accuracy: {acc:.4f}")

    # Classification Report
    st.text("Classification Report:")
    st.text(classification_report(y_test, y_pred))

    # Confusion Matrix
    cm = confusion_matrix(y_test, y_pred)
    fig, ax = plt.subplots()
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', ax=ax)
    ax.set_title("Confusion Matrix")
    ax.set_xlabel("Predicted")
    ax.set_ylabel("Actual")
    st.pyplot(fig)

    # Visualizations
    st.subheader("Feature Distributions")
    numeric_cols = df.select_dtypes(include=['int64', 'float64']).columns
    selected_col = st.selectbox("Select feature for boxplot", numeric_cols)
    fig2, ax2 = plt.subplots()
    sns.boxplot(x='HeartDisease', y=selected_col, data=df, ax=ax2)
    st.pyplot(fig2)

    # Prediction form
    st.subheader("🔍 Make a Prediction")
    input_data = {}
    for col in numeric_features:
        input_data[col] = st.number_input(f"{col}", value=float(X[col].mean()))
    for col in categorical_features:
        input_data[col] = st.selectbox(f"{col}", options=df[col].unique())

    if st.button("Predict Heart Disease"):
        input_df = pd.DataFrame([input_data])
        if model_name != 'Neural Network (Keras)':
            pred = clf.predict(input_df)[0]
        else:
            input_processed = preprocessor.transform(input_df)
            pred_prob = model.predict(input_processed)
            pred = int((pred_prob > 0.5).astype("int32")[0][0])
        st.success(f"Prediction: {'Heart Disease' if pred == 1 else 'No Heart Disease'}")

else:
    st.info("Awaiting CSV file upload.")
