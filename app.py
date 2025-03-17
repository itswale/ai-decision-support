import streamlit as st
import pandas as pd
import numpy as np
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, precision_score, recall_score
from sklearn.preprocessing import OneHotEncoder
import plotly.express as px
import lime.lime_tabular
import matplotlib.pyplot as plt
import pdfplumber
import io
import base64
import time

# Favicon ("P" shape, base64-encoded)
favicon = "data:image/png;base64,iVBORw0KGgoAAAANSUhEUgAAACAAAAAgCAYAAABzenr0AAAABmJLR0QA/wD/AP+gvaeTAAABgElEQVRYw+2UsUrDUBSG/6sQvIMXsLCwsLCwsLDRsLCwsLDQ2NiLh1uwcLF3iI2F/4CFhYWFhYXFlsFycXGwD+w5zM5OkrzZ3eSmblnZ2dm5RwD+AH+AC8Af4Abwfr8fr9frwW63G5/P5/P5fD6fz2azWTMMA7/fj8fj0Wg0YrVaMZ/PJ5PJYrFYLBaz2SwWi8ViMpvNYrFYLEnT9Hq9XC6Xy+VyOaLRaDQej8fj8Xg8Ho/H4/F4PB6Px+Pxer1eLpfL5XK5XC6Xy+VyOaLRaDQej8fj8Xg8Ho/H4/F4PB6Px+Pxer1eLpfL5XK5XC6Xy+VyOaLRaDQej8fj8Xg8Ho/H4/F4PB6Px+Pxer1eLpfL5XK5XC6Xy+VyOaLRaDQej8fj8Xg8Ho/H4/F4PB6Px+Pxer1eLpfL5XK5XC6Xy+VyuVzOZrNYLBaz2SwWi8ViMpvNYrFYLEn/AKSsJ9t6nHeQAAAAAElFTkSuQmCC"

st.set_page_config(page_title="AI Decision Support App", page_icon=favicon)

# Demo datasets
DEMO_DATA = {
    "Finance": pd.DataFrame({
        'CustomerID': ['CUST001', 'CUST002', 'CUST003', 'CUST004', 'CUST005', 'CUST006'],
        'Income': [45000.0, 25000.0, 60000.0, 30000.0, 70000.0, 20000.0],
        'CreditScore': [720, 580, 800, 620, 750, 540],
        'LoanAmount': [10000.0, 15000.0, 5000.0, 12000.0, 8000.0, 20000.0],
        'EmploymentStatus': ['Employed', 'Unemployed', 'Self-Employed', 'Employed', 'Employed', 'Unemployed'],
        'Default': ['No', 'Yes', 'No', 'Yes', 'No', 'Yes']
    }),
    "Retail": pd.DataFrame({
        'StoreID': ['STORE01', 'STORE02', 'STORE03', 'STORE04', 'STORE05', 'STORE06'],
        'DailySales': [500.0, 300.0, 700.0, 400.0, 800.0, 200.0],
        'FootTraffic': [150, 80, 200, 120, 250, 60],
        'PromoActive': ['Yes', 'No', 'Yes', 'No', 'Yes', 'No'],
        'Demand': ['High', 'Low', 'High', 'Low', 'High', 'Low']
    }),
    "Healthcare": pd.DataFrame({
        'PatientID': ['PAT001', 'PAT002', 'PAT003', 'PAT004', 'PAT005', 'PAT006'],
        'Age': [65, 45, 72, 55, 30, 80],
        'BloodPressure': [140, 120, 150, 130, 110, 160],
        'Smoker': ['Yes', 'No', 'Yes', 'No', 'No', 'Yes'],
        'Risk': ['High', 'Low', 'High', 'Low', 'Low', 'High']
    })
}

# Initialize/reset session state
def reset_state(mode="Finance"):
    st.session_state.clear()
    st.session_state.model = None
    st.session_state.encoder = None
    st.session_state.feature_names = None
    st.session_state.uploaded_file = None
    st.session_state.mode = mode
    st.session_state.features = []
    st.session_state.target = None
    st.session_state.section = "start"
    st.session_state.data_source = f"{mode} Demo ({'Loan Defaults' if mode == 'Finance' else 'Sales Demand' if mode == 'Retail' else 'Patient Risk'})"

if 'model' not in st.session_state:
    reset_state()

# Sidebar setup
with st.sidebar:
    st.markdown("### Settings")
    mode = st.selectbox("Mode", ["Finance", "Retail", "Healthcare"], index=["Finance", "Retail", "Healthcare"].index(st.session_state.mode), help="Choose your use case.")
    if mode != st.session_state.mode:
        reset_state(mode)
        st.rerun()
    
    data_source = st.selectbox(
        "Data Source",
        ["Finance Demo (Loan Defaults)", "Retail Demo (Sales Demand)", "Healthcare Demo (Patient Risk)", "Upload Your Own"],
        index=["Finance Demo (Loan Defaults)", "Retail Demo (Sales Demand)", "Healthcare Demo (Patient Risk)", "Upload Your Own"].index(st.session_state.data_source),
        help="Use a demo or upload data."
    )
    if data_source != st.session_state.data_source:
        st.session_state.target = None
        st.session_state.data_source = data_source
    
    uploaded_file = None
    if data_source == "Upload Your Own":
        file_type = st.radio("File Type", ("CSV", "PDF"))
        uploaded_file = st.file_uploader(f"Upload {file_type}", type=[file_type.lower()])
    if uploaded_file != st.session_state.uploaded_file:
        st.session_state.uploaded_file = uploaded_file

    if st.button("Reset"):
        with st.spinner("Resetting app..."):
            time.sleep(2)
            reset_state(mode)
            st.success("App reset successfully!")
        st.rerun()

# Data loading with mode alignment check
data = None
mode_dataset_map = {"Finance": "Finance Demo (Loan Defaults)", "Retail": "Retail Demo (Sales Demand)", "Healthcare": "Healthcare Demo (Patient Risk)"}
mode_targets = {"Finance": "Default", "Retail": "Demand", "Healthcare": "Risk"}
if data_source in mode_dataset_map.values():
    expected_mode = next(k for k, v in mode_dataset_map.items() if v == data_source)
    if mode != expected_mode:
        st.error(f"Mode mismatch! For '{data_source}', please select '{expected_mode}' mode in the sidebar.")
    else:
        data = DEMO_DATA[mode]
elif uploaded_file:
    try:
        data = pd.read_csv(uploaded_file) if file_type == "CSV" else pd.DataFrame(pdfplumber.open(uploaded_file).pages[0].extract_table()[1:], columns=pdfplumber.open(uploaded_file).pages[0].extract_table()[0])
        expected_target = mode_targets[mode]
        if expected_target not in data.columns:
            st.error(f"Target '{expected_target}' not found in your dataset. For '{mode}' mode, use a dataset with a '{expected_target}' column, or switch to a matching mode (e.g., {'Finance' if expected_target != 'Default' else 'Retail'} for '{mode_targets.get('Finance' if expected_target != 'Default' else 'Retail')}').")
            data = None
    except Exception as e:
        st.error(f"Upload failed: {e}. Please upload a valid {file_type} file compatible with '{mode}' mode (e.g., including '{mode_targets[mode]}' column).")
        data = None

if data is not None:
    data.index = data.index + 1
    for col in ['Income', 'CreditScore', 'LoanAmount', 'DailySales', 'FootTraffic', 'Age', 'BloodPressure']:
        if col in data:
            data[col] = pd.to_numeric(data[col], errors='coerce')
    data = data.dropna()

    # Feature and target selection with mode alignment
    all_columns = data.columns.tolist()
    features = st.sidebar.multiselect("Features", all_columns, default=[col for col in all_columns if col not in ['CustomerID', 'StoreID', 'PatientID', 'Default', 'Demand', 'Risk']], help="Variables to analyze.")
    st.session_state.features = features
    
    default_target = mode_targets.get(mode, all_columns[0])
    try:
        target = st.sidebar.selectbox("Target", all_columns, index=all_columns.index(st.session_state.target if st.session_state.target in all_columns else default_target), help="What to predict (2-3 values best).")
    except ValueError:
        st.error(f"Target '{default_target}' not found in your dataset. For '{mode}' mode, ensure your dataset includes '{default_target}', or select a different mode that matches your data (e.g., {'Finance' if default_target != 'Default' else 'Retail'} for '{mode_targets.get('Finance' if default_target != 'Default' else 'Retail')}').")
        target = st.sidebar.selectbox("Target", all_columns, index=0, help="What to predict (2-3 values best).")
    st.session_state.target = target
    
    with st.sidebar.form("train_form"):
        train_button = st.form_submit_button("Train Model")
        if train_button:
            st.session_state.section = "results"
    
    st.sidebar.info(f"Ready! Pick features (e.g., {', '.join(all_columns[:2])}) and target (e.g., {default_target}).")

# Main app
st.title("AI Decision Support App")
if data is not None and features and st.session_state.target and st.session_state.target not in features:
    try:
        # Dataset preview
        st.subheader("Dataset Preview (Rows start at 1)", anchor="preview")
        rows_per_page = st.number_input("Rows per page", 1, len(data), 5, step=1)
        page = st.number_input("Page", 1, (len(data) // rows_per_page) + 1, 1, step=1)
        start_idx, end_idx = (page - 1) * rows_per_page, (page - 1) * rows_per_page + rows_per_page
        st.dataframe(data.iloc[start_idx:end_idx])
        st.download_button("Export Dataset as CSV", data.to_csv(index=True), f"{mode.lower()}_data.csv", "text/csv")

        # Data prep
        X = data[features]
        y = data[st.session_state.target]
        n_classes = len(np.unique(y))
        if n_classes < 2:
            st.error(f"Target '{st.session_state.target}' needs 2+ unique values to train a model.")
        elif n_classes > 3:
            st.warning(f"Target '{st.session_state.target}' has {n_classes} values. Binary targets work best.")
        if len(data) < 50:
            st.warning("Small dataset (<50 rows) may lead to less reliable predictions. Confidence scores may vary.")
        
        numeric_cols = X.select_dtypes(exclude=['object']).columns
        for col in numeric_cols:
            if X[col].std() == 0:
                st.warning(f"Feature '{col}' has no variation (all {X[col].iloc[0]}).")
        
        categorical_cols = X.select_dtypes(include=['object']).columns
        categorical_indices = [X.columns.get_loc(col) for col in categorical_cols]
        if categorical_cols.any():
            encoder = OneHotEncoder(sparse_output=False, drop='first', handle_unknown='ignore')
            X_encoded = pd.DataFrame(encoder.fit_transform(X[categorical_cols]), index=X.index)
            X_encoded.columns = encoder.get_feature_names_out(categorical_cols)
            X = pd.concat([X.drop(columns=categorical_cols), X_encoded], axis=1)
            st.session_state.encoder = encoder
        else:
            st.session_state.encoder = None

        if X.empty or y.empty:
            st.error("Select at least one feature and a valid target.")
            st.stop()

        # Model training
        if train_button:
            with st.spinner("Training model..."):
                time.sleep(3)
                model = RandomForestClassifier(random_state=42, n_estimators=100, min_samples_split=5)
                model.fit(X, y)
                st.session_state.model = model
                st.session_state.feature_names = X.columns.tolist()
            st.success("Model trained successfully!")

        if st.session_state.model:
            model = st.session_state.model
            y_pred = model.predict(X)

            # Visualization
            st.subheader("Feature vs. Target Visualization", anchor="visualization")
            if numeric_cols.any():
                plot_feature = st.selectbox("Select numeric feature", numeric_cols)
                fig = px.box(data, x=st.session_state.target, y=plot_feature, title=f"{plot_feature} vs. {st.session_state.target}")
                st.plotly_chart(fig)
            else:
                st.info("No numeric features to plot.")

            # Performance metrics
            st.subheader("Model Performance", anchor="performance")
            accuracy = accuracy_score(y, y_pred)
            st.write(f"Accuracy: {accuracy:.2f}")
            average = 'binary' if n_classes == 2 else 'weighted'
            precision = precision_score(y, y_pred, pos_label=np.unique(y)[1] if n_classes == 2 else None, average=average, zero_division=0)
            recall = recall_score(y, y_pred, pos_label=np.unique(y)[1] if n_classes == 2 else None, average=average, zero_division=0)
            st.write(f"Precision: {precision:.2f}")
            st.write(f"Recall: {recall:.2f}")
            if accuracy > 0.9:
                st.warning("High scores may indicate overfitting. Test predictions with real-world inputs.")
            pred_df = pd.DataFrame({'Actual': y, 'Predicted': y_pred}, index=data.index)
            st.download_button("Export Predictions as CSV", pred_df.to_csv(index=True), f"{mode.lower()}_predictions.csv", "text/csv")

            # Feature importance
            if hasattr(model, 'feature_importances_'):
                fig = px.bar(x=X.columns, y=model.feature_importances_, title="Feature Importance")
                st.plotly_chart(fig)

            # LIME explanation
            st.subheader("Prediction Explanation (First Row)", anchor="explanation")
            with st.spinner("Generating explanation..."):
                try:
                    explainer = lime.lime_tabular.LimeTabularExplainer(
                        X.values, feature_names=X.columns.tolist(), class_names=np.unique(y).tolist(),
                        categorical_features=categorical_indices, mode='classification', random_state=42
                    )
                    exp = explainer.explain_instance(X.iloc[0].values, model.predict_proba, num_features=5)
                    fig = exp.as_pyplot_figure()
                    plt.tight_layout()
                    st.pyplot(fig, clear_figure=True)
                    buf = io.BytesIO()
                    fig.savefig(buf, format="png", bbox_inches="tight", dpi=100)
                    plt.close(fig)
                    st.download_button("Export Explanation as PNG", buf.getvalue(), f"{mode.lower()}_lime_explanation.png", "image/png")
                    st.markdown("**Explanation**: Top features driving the first row's prediction.")
                except Exception as e:
                    st.error(f"Explanation failed: {e}. Check for feature variation or try a different dataset.")

        # Prediction section with real-world logic
        st.subheader("Make a Prediction", anchor="prediction")
        with st.form("predict_form"):
            inputs = {}
            for col in features:
                if col in categorical_cols:
                    options = data[col].unique().tolist()
                    inputs[col] = st.selectbox(f"{col}", options, index=0 if options else None)
                else:
                    inputs[col] = st.number_input(f"{col}", value=float(data[col].mean()) if col in data else 0.0, step=0.01)
            predict_button = st.form_submit_button("Predict")
            if predict_button:
                st.session_state.section = "prediction"

        if predict_button:
            if not st.session_state.model:
                st.error("No model trained yet! Please click 'Train Model' in the sidebar first.")
            else:
                with st.spinner("Predicting..."):
                    time.sleep(2)
                    try:
                        input_df = pd.DataFrame([inputs])
                        if categorical_cols.any() and st.session_state.encoder:
                            input_encoded = pd.DataFrame(st.session_state.encoder.transform(input_df[categorical_cols]))
                            input_encoded.columns = st.session_state.encoder.get_feature_names_out(categorical_cols)
                            input_df = pd.concat([input_df.drop(columns=categorical_cols), input_encoded], axis=1)
                        missing_cols = [col for col in st.session_state.feature_names if col not in input_df.columns]
                        for col in missing_cols:
                            input_df[col] = 0
                        input_df = input_df[st.session_state.feature_names]
                        pred = model.predict(input_df)[0]
                        pred_proba = model.predict_proba(input_df)[0]
                        confidence = max(pred_proba) * 100
                        # Probability of "No" (e.g., no default, high demand, low risk)
                        proba_no = pred_proba[np.where(model.classes_ == 'No')[0][0]] if 'No' in model.classes_ else pred_proba[np.where(model.classes_ == 'Low')[0][0]] if 'Low' in model.classes_ else 1 - pred_proba[0]

                        expected_target = mode_targets[mode]
                        if st.session_state.target != expected_target:
                            st.warning(f"Target '{st.session_state.target}' doesn't match '{mode}' mode's expected target '{expected_target}'. Predictions may not align with mode-specific messages.")

                        if mode == "Finance" and st.session_state.target == "Default":
                            loan_amount = inputs.get('LoanAmount', 10000.0)
                            if pred == "No" and proba_no > 0.7:
                                pred_msg = f"Approve loan? Yes - Suggested amount: ${loan_amount:.2f} (Confidence: {confidence:.1f}%)"
                            elif pred == "No" and proba_no > 0.5:
                                pred_msg = f"Approve loan? Yes with caution - Suggested amount: ${loan_amount:.2f} (Moderate confidence: {confidence:.1f}%)"
                            else:
                                pred_msg = f"Approve loan? No - Consult risk assessment (Confidence: {confidence:.1f}%)"
                        elif mode == "Retail" and st.session_state.target == "Demand":
                            if pred == "High" and proba_no < 0.3:  # High demand = low "No" probability
                                pred_msg = f"Predicted demand: High - Stock heavily (Confidence: {confidence:.1f}%)"
                            elif pred == "High":
                                pred_msg = f"Predicted demand: High - Stock moderately (Moderate confidence: {confidence:.1f}%)"
                            else:
                                pred_msg = f"Predicted demand: Low - Stock lightly (Confidence: {confidence:.1f}%)"
                        elif mode == "Healthcare" and st.session_state.target == "Risk":
                            if pred == "High" and proba_no < 0.3:  # High risk = low "No" probability
                                pred_msg = f"Patient risk: High - Urgent care recommended (Confidence: {confidence:.1f}%)"
                            elif pred == "High":
                                pred_msg = f"Patient risk: High - Monitor closely (Moderate confidence: {confidence:.1f}%)"
                            else:
                                pred_msg = f"Patient risk: Low - Routine checkup sufficient (Confidence: {confidence:.1f}%)"
                        else:
                            pred_msg = f"Predicted {st.session_state.target}: {pred} (Confidence: {confidence:.1f}%)"
                        st.success(f"Prediction completed! {pred_msg}")
                        if len(data) < 50:
                            st.info("Note: Small dataset predictions may be less reliable. Confidence reflects model certainty, but more data improves accuracy.")
                    except Exception as e:
                        st.error(f"Prediction failed: {e}. Ensure your inputs and dataset align with '{mode}' mode (e.g., including '{mode_targets[mode]}' as target).")

    except Exception as e:
        st.error(f"Oops! Something went wrong: {e}. Please check your dataset for '{mode}' mode compatibility (e.g., including '{mode_targets[mode]}' column).")
elif data is not None:
    st.info("Select features and a target.")
else:
    st.info("Choose a data source or upload a file.")

# Scroll to section
if st.session_state.section != "start":
    st.markdown(f'<script>document.getElementById("{st.session_state.section}").scrollIntoView();</script>', unsafe_allow_html=True)

# Footer
with st.expander("About AI Decision Support App"):
    st.markdown("""
    **AI Decision Support App**: Decision-making made simple. Train models, get insights, and export results.

    - **Finance**: Loan default risk (needs 'Default' column).
    - **Retail**: Sales demand planning (needs 'Demand' column).
    - **Healthcare**: Patient risk assessment (needs 'Risk' column).
    """)

# Styling
st.markdown("""
<style>
h2 { color: #1E90FF; }
.sidebar .sidebar-content { background-color: #f8f9fa; padding: 10px; border-radius: 5px; }
</style>
""", unsafe_allow_html=True)