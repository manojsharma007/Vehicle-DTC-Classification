
import streamlit as st
import pandas as pd
import numpy as np
import joblib
import matplotlib.pyplot as plt
import seaborn as sns

from sklearn.metrics import (
    accuracy_score,
    precision_score,
    recall_score,
    f1_score,
    matthews_corrcoef,
    confusion_matrix
)

models_loaded = False
models = None
scaler = None
label_encoder = None

st.set_page_config(
    page_title="Vehicle DTC Classification",
    page_icon="🚗",
    layout="wide",
    initial_sidebar_state="expanded"
)

@st.cache_resource
def load_models():
    try:
        models = {
            "Logistic Regression": joblib.load("model/logistic_regression.pkl"),
            "Decision Tree": joblib.load("model/decision_tree.pkl"),
            "KNN": joblib.load("model/knn.pkl"),
            "Naive Bayes": joblib.load("model/naive_bayes.pkl"),
            "Random Forest": joblib.load("model/random_forest.pkl"),
            "XGBoost (Best)": joblib.load("model/xgboost.pkl")
        }

        scaler = joblib.load("model/scaler.pkl")
        label_encoder = joblib.load("model/label_encoder.pkl")

        return models, scaler, label_encoder, True

    except Exception:
        return None, None, None, False


models, scaler, label_encoder, models_loaded = load_models()

st.markdown("""
<style>
.big-title {
    font-size: 32px;
    font-weight: 700;
}
.sub-title {
    font-size: 16px;
    color: #555;
}
.metric-card {
    background-color:#f8f9fa;
    padding:15px;
    border-radius:10px;
    text-align:center;
    box-shadow:0px 0px 6px rgba(0,0,0,0.05);
}
</style>
""", unsafe_allow_html=True)

st.markdown('<div class="big-title">🚗 Vehicle Diagnostic Trouble Code Classification</div>', unsafe_allow_html=True)
st.markdown(
    '<div class="sub-title">'
    'Evaluation of machine learning models using uploaded labeled test data '
    'from OBD-II vehicle sensor signals.'
    '</div>',
    unsafe_allow_html=True
)

st.markdown("---")

st.sidebar.header("⚙️ Input Controls")
# -------------------------------
# Download sample test dataset
# -------------------------------
st.sidebar.markdown("### 📥 Download Sample Test Data")

try:
    sample_test_df = pd.read_csv("data/test_data_dtc_labeled.csv")

    st.sidebar.download_button(
        label="⬇️ Download Labeled Test CSV",
        data=sample_test_df.to_csv(index=False),
        file_name="test_data_dtc_labeled.csv",
        mime="text/csv"
    )

except Exception:
    st.sidebar.info("ℹ️ Sample test dataset not available for download.")


if models_loaded:
    st.sidebar.success("✅ Models loaded successfully")
else:
    st.sidebar.warning("⚠️ Models not loaded")
model_choice = st.sidebar.selectbox(
    "🔽 Select Model",
    [
        "XGBoost (Best)",
        "Random Forest",
        "Decision Tree",
        "KNN",
        "Logistic Regression",
        "Naive Bayes"
    ]
)

st.sidebar.markdown("### 📤 Upload Test Dataset")

uploaded_file = st.sidebar.file_uploader(
    "Upload LABELED Test Dataset (CSV / XLSX)",
    type=["csv", "xlsx"]
)


left_col, right_col = st.columns([2, 3])

with left_col:
    st.subheader("📊 Model Performance Overview")
    st.info("Evaluation metrics computed on the uploaded labeled test dataset.")
with right_col:
    st.subheader("🔥 Confusion Matrix")
    st.info("Confusion matrix based on actual vs predicted DTC classes.")


expected_features = [
    'Barometric Pressure',
    'Divers Demand Engine Percent Torque',
    'Relative Throttle Position',
    'Accelerator Pedal Position E',
    'Consumption Rate',
    'Load',
    'Mass Air Flow',
    'Speed',
    'Ambient Air Temperature',
    'RPM',
    'Actual Engine Percent Torque',
    'Fuel Level',
    'Accelerator Pedal Position D',
    'Throttle Position'
]

test_df = None
y_true_labels = None
if uploaded_file is not None:
    try:
        if uploaded_file.name.lower().endswith(".csv"):
            raw_df = pd.read_csv(uploaded_file, sep=",")
            if raw_df.shape[1] == 1:
                raw_df = pd.read_csv(uploaded_file, sep=";")
        else:
            raw_df = pd.read_excel(uploaded_file, engine="openpyxl")

        raw_df.columns = (
            raw_df.columns
            .str.strip()
            .str.lower()
            .str.replace('%', '', regex=False)
            .str.replace('(', '', regex=False)
            .str.replace(')', '', regex=False)
        )

        feature_map = {
            'barometric pressure': 'Barometric Pressure',
            'divers demand engine percent torque': 'Divers Demand Engine Percent Torque',
            'relative throttle position': 'Relative Throttle Position',
            'accelerator pedal position e': 'Accelerator Pedal Position E',
            'consumption rate': 'Consumption Rate',
            'load': 'Load',
            'mass air flow': 'Mass Air Flow',
            'speed': 'Speed',
            'ambient air temperature': 'Ambient Air Temperature',
            'rpm': 'RPM',
            'actual engine percent torque': 'Actual Engine Percent Torque',
            'fuel level': 'Fuel Level',
            'accelerator pedal position d': 'Accelerator Pedal Position D',
            'throttle position': 'Throttle Position'
        }

        missing = set(feature_map.keys()) - set(raw_df.columns)

        if missing:
            st.error(f"❌ Missing feature columns: {list(missing)}")
        elif "dtc_code" not in raw_df.columns:
            st.error("❌ Missing target column: dtc_code")
        else:
            test_df = raw_df[list(feature_map.keys())].rename(columns=feature_map)
            y_true_labels = raw_df["dtc_code"].values

            st.subheader("📄 Uploaded Test Dataset Preview")
            st.dataframe(raw_df.head())

    except Exception as e:
        st.error("❌ Error reading uploaded dataset")
        st.exception(e)


if test_df is not None and y_true_labels is not None and models_loaded:

    # ---- Evaluation Context Banner ----
    st.markdown(
        f"""
        <div style="
            background-color:#eef2f7;
            padding:12px;
            border-radius:8px;
            margin-bottom:15px;
        ">
        <b>Evaluation Context</b><br>
        Model: <b>{model_choice}</b> &nbsp; | &nbsp;
        Test Samples: <b>{len(test_df)}</b> &nbsp; | &nbsp;
        Classes: <b>{len(np.unique(y_true_labels))}</b>
        </div>
        """,
        unsafe_allow_html=True
    )
    y_true = label_encoder.transform(y_true_labels)
    model = models[model_choice]

    if model_choice in ["Logistic Regression", "KNN", "XGBoost (Best)"]:
        X_eval = scaler.transform(test_df)
    else:
        X_eval = test_df.values

    y_pred = model.predict(X_eval)

    with left_col:
        acc = accuracy_score(y_true, y_pred)
        prec = precision_score(y_true, y_pred, average="macro")
        rec = recall_score(y_true, y_pred, average="macro")
        f1 = f1_score(y_true, y_pred, average="macro")
        mcc = matthews_corrcoef(y_true, y_pred)

        metric_cols = st.columns(5)
        metrics = [
            ("Accuracy", acc),
            ("Precision", prec),
            ("Recall", rec),
            ("F1-score", f1),
            ("MCC", mcc)
        ]

        for col, (label, value) in zip(metric_cols, metrics):
            col.markdown(
                f"""
                <div class="metric-card">
                    <div style="font-size:14px;color:#666">{label}</div>
                    <div style="font-size:22px;font-weight:700">{value:.3f}</div>
                </div>
                """,
                unsafe_allow_html=True
            )


    with right_col:
        st.markdown(
            """
            <div style="
                background-color:#ffffff;
                padding:10px;
                border-radius:10px;
                box-shadow:0px 0px 6px rgba(0,0,0,0.05);
            ">
            """,
            unsafe_allow_html=True
        )

        cm = confusion_matrix(y_true, y_pred)

        fig, ax = plt.subplots(figsize=(6, 5))
        sns.heatmap(
            cm,
            annot=True,
            fmt="d",
            cmap="Blues",
            xticklabels=label_encoder.classes_,
            yticklabels=label_encoder.classes_,
            ax=ax
        )
        ax.set_xlabel("Predicted")
        ax.set_ylabel("Actual")
        st.pyplot(fig)

        st.markdown("</div>", unsafe_allow_html=True)

st.markdown(
    """
    <hr>
    <div style="text-align:center;color:#888;font-size:13px;">
    Vehicle DTC Classification | Streamlit App | BITS ML Assignment
    </div>
    """,
    unsafe_allow_html=True
)
