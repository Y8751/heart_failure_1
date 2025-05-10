
import streamlit as st
import pandas as pd
import joblib
import seaborn as sns
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix

# العنوان

st.title("تطبيق تصنيف أمراض القلب")

# تحميل الملف CSV

uploaded_file = st.file_uploader("رفع ملف CSV الخاص بك", type=['csv'])

@st.cache_data
def load_data(file):
    df = pd.read_csv(file)
    return df

# المعالجة المبدئية

@st.cache_data
def preprocess_data(df):
    # معالجة البيانات (مثل التحجيم، الترميز، الخ)
    X = df.drop('HeartDisease', axis=1)
    y = df['HeartDisease']
    return X, y

# تحميل النماذج المدربة مسبقًا

def load_models():
    model_names = ['Logistic Regression', 'KNN', 'Random Forest']  # على سبيل المثال، النماذج الأفضل بناءً على الـ recall
    models = {}
    for name in model_names:
        models[name] = joblib.load(f"{name}_model.pkl")  # تأكد من المسار الصحيح للنماذج
    return models

# دالة التنبؤ

def make_prediction(model, input_data, preprocessor):
    input_processed = preprocessor.transform(input_data)
    return model.predict(input_processed)

if uploaded_file is not None:
    df = load_data(uploaded_file)
    st.write("### معاينة البيانات", df.head())
    X, y = preprocess_data(df)

    # تحميل أفضل 3 نماذج (تم تدريبها مسبقًا)
    models = load_models()

    # اختيار نموذج للتنبؤ
    model_name = st.selectbox("اختر نموذجًا", list(models.keys()))
    model = models[model_name]

    # نموذج التنبؤ
    st.subheader("🔍 إجراء التنبؤ")
    input_data = {}
    for col in X.columns:
        input_data[col] = st.number_input(f"{col}", value=float(X[col].mean()))

    if st.button("تنبؤ بأمراض القلب"):
        input_df = pd.DataFrame([input_data])
        pred = make_prediction(model, input_df, preprocessor)
        st.success(f"التنبؤ: {'أمراض القلب' if pred == 1 else 'لا توجد أمراض قلبية'}")

    # أداء النموذج ومصفوفة الالتباس (اختياري)
    y_pred = model.predict(X)  # استخدام النموذج المختار لإجراء التنبؤات
    acc = accuracy_score(y, y_pred)
    st.success(f"دقة النموذج: {acc:.4f}")
    st.text("تقرير التصنيف:")
    st.text(classification_report(y, y_pred))

    # مصفوفة الالتباس
    cm = confusion_matrix(y, y_pred)
    fig, ax = plt.subplots()
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', ax=ax)
    ax.set_title("مصفوفة الالتباس")
    ax.set_xlabel("المتنبأ به")
    ax.set_ylabel("الفعلي")
    st.pyplot(fig)

else:
    st.info("في انتظار تحميل ملف CSV.")
