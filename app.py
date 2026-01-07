import streamlit as st
import pandas as pd
import numpy as np
import joblib
import shap
import dice_ml
import matplotlib.pyplot as plt
from sklearn.preprocessing import MinMaxScaler
import warnings

warnings.filterwarnings('ignore')

# ==========================================
# 0. Setup & Styling
# ==========================================
st.set_page_config(page_title="Credit Risk XAI System", layout="wide")

# Custom CSS for better UI
st.markdown("""
<style>
    /* Sidebar styling */
    [data-testid="stSidebar"] { width: 600px !important; background-color: #f5f5f5; }
    
    /* Main content area */
    .main { background-color: #ffffff; }
    
    /* Header styling */
    h1 { color: #1f77b4; font-size: 2.5rem; margin-bottom: 0.5rem; }
    h2 { color: #2c3e50; font-size: 1.8rem; margin-top: 1.5rem; }
    h3 { color: #34495e; font-size: 1.3rem; }
    
    /* Metric styling */
    [data-testid="metric-container"] { background-color: #f8f9fa; padding: 1rem; border-radius: 8px; }
    
    /* Card styling */
    .card { background-color: #f8f9fa; padding: 1.5rem; border-radius: 10px; border-left: 4px solid #1f77b4; margin-bottom: 1rem; }
    
    /* Success/Error styling */
    [data-testid="stSuccess"] { background-color: #d4edda; }
    [data-testid="stError"] { background-color: #f8d7da; }
    [data-testid="stWarning"] { background-color: #fff3cd; }
    [data-testid="stInfo"] { background-color: #d1ecf1; }
    
    /* Button styling */
    button { background-color: #1f77b4 !important; color: white !important; }
    
    /* Divider */
    hr { border-color: #e0e0e0; margin: 2rem 0; }
</style>
""", unsafe_allow_html=True)

# ==========================================
# 1. Load Model & Data
# ==========================================
@st.cache_resource
def load_all():
    model = joblib.load('xgb_model.pkl')
    X_train = joblib.load('X_train.pkl')
    X_test = joblib.load('X_test.pkl')
    return model, X_train, X_test

model, X_train, X_test = load_all()

# ==========================================
# 2. ModelWrapper for DiCE
# ==========================================
class ModelWrapper:
    def __init__(self, mdl):
        self.mdl = mdl
    def predict(self, X):
        return self.mdl.predict(X.apply(pd.to_numeric, errors='coerce').fillna(0))
    def predict_proba(self, X):
        return self.mdl.predict_proba(X.apply(pd.to_numeric, errors='coerce').fillna(0))

wrapped_model = ModelWrapper(model)

# ==========================================
# 3. Helper: Preprocess Input
# ==========================================
def preprocess_input(raw_dict):
    """
    Convert raw input (Age, Duration, Credit amount, etc.) to model features
    Mirrors the assignment.ipynb feature engineering pipeline
    """
    df_raw = pd.DataFrame([raw_dict])
    
    # Mathematical Transformations
    df_raw["Monthly pay"] = (df_raw["Credit amount"] / df_raw["Duration"])
    df_raw["Credit amount_sq"] = df_raw["Credit amount"] ** 2
    
    # Age Binning
    cat_age = pd.Series(dtype='object', index=df_raw.index)
    cat_age[df_raw["Age"] < 25] = "0-25"
    cat_age[((df_raw["Age"] >= 25) & (df_raw["Age"] < 30))] = "25-30"
    cat_age[((df_raw["Age"] >= 30) & (df_raw["Age"] < 35))] = "30-35"
    cat_age[((df_raw["Age"] >= 35) & (df_raw["Age"] < 40))] = "35-40"
    cat_age[((df_raw["Age"] >= 40) & (df_raw["Age"] < 50))] = "40-50"
    cat_age[((df_raw["Age"] >= 50) & (df_raw["Age"] < 76))] = "50-75"
    df_raw["Age"] = cat_age
    
    # Duration Binning
    def categorize_duration(i):
        if i < 12: return "0-12"
        elif (i >= 12) and (i < 24): return "12-24"
        elif (i >= 24) and (i < 36): return "24-36"
        elif (i >= 36) and (i < 48): return "36-48"
        elif (i >= 48) and (i < 60): return "48-60"
        elif (i >= 60) and (i <= 72): return "60-72"
        return str(i)
    df_raw["Duration"] = df_raw["Duration"].apply(categorize_duration)
    
    # Job Classification (convert numbers to text)
    job_map = {0: "unskilled", 1: "resident", 2: "skilled", 3: "highly skilled"}
    df_raw["Job"] = df_raw["Job"].replace(job_map)
    
    # One-Hot Encoding
    target_col = "Risk"
    exclude_cols = [target_col, "Credit amount", "Monthly pay", "Credit amount_sq"]
    categorical_cols = [c for c in df_raw.columns if c not in exclude_cols]
    
    df_encoded = pd.get_dummies(df_raw, columns=categorical_cols, prefix=categorical_cols)
    
    # Scaling using training ranges to match model preprocessing
    scale_cols = ["Credit amount", "Monthly pay", "Credit amount_sq"]
    for col in scale_cols:
        if (col in X_train.columns) and (col in df_encoded.columns):
            min_v = X_train[col].min()
            max_v = X_train[col].max()
            if max_v > min_v:
                df_encoded[col] = (df_encoded[col] - min_v) / (max_v - min_v)
            else:
                df_encoded[col] = 0.0
    
    # Ensure all model columns are present
    for col in X_train.columns:
        if col not in df_encoded.columns:
            df_encoded[col] = 0
    
    # Reorder columns to match model training
    df_encoded = df_encoded[X_train.columns]
    
    return df_encoded

# ==========================================
# 4. Main UI
# ==========================================
st.title("Credit Risk Assessment System")

# Sidebar: User Input
st.sidebar.title("Applicant Information")
st.sidebar.markdown("Enter applicant details below")
st.sidebar.markdown("---")

# Organize inputs in columns
col1, col2 = st.sidebar.columns(2)

with col1:
    age = st.number_input("Age", min_value=18, max_value=100, value=35, key="age")
    credit_amount = st.number_input("Credit Amount (EUR)", min_value=100, max_value=20000, value=3000, key="credit_amount")
    job = st.selectbox("Job Level", options=[0, 1, 2, 3], 
                       format_func=lambda x: ["Unskilled", "Resident", "Skilled", "Highly Skilled"][x], key="job")
    savings = st.selectbox("Savings Account", 
                          options=["<100 DM", "100-500 DM", "500-1000 DM", ">1000 DM", "NA"], key="savings")

with col2:
    duration = st.number_input("Duration (months)", min_value=1, max_value=72, value=12, key="duration")
    sex = st.selectbox("Sex", options=["male", "female"], key="sex")
    housing = st.selectbox("Housing Type", options=["free", "own", "rent"], key="housing")
    status = st.selectbox("Checking Account", 
                         options=["little", "moderate", "rich", "NA"], key="status")

st.sidebar.markdown("---")

# Purpose in full width
purpose = st.sidebar.selectbox("Purpose", 
                      options=["car", "furniture/equipment", "education", "appliances", "repairs", 
                              "business", "vacation", "radio/TV", "training", "other"], key="purpose")

# Build input dict
raw_input = {
    'Age': age,
    'Credit amount': credit_amount,
    'Duration': duration,
    'Job': job,
    'Sex': sex,
    'Housing': housing,
    'Saving accounts': savings,
    'Checking account': status,
    'Purpose': purpose
}

# Preprocess
try:
    sample_processed = preprocess_input(raw_input)
except Exception as e:
    st.error(f"Preprocessing error: {e}")
    st.stop()

# ==========================================
# 5. Prediction
# ==========================================
st.markdown("---")
col1, col2 = st.columns([1.2, 1.8])

with col1:
    st.subheader("Prediction Result")
    prob = model.predict_proba(sample_processed)[0][1]
    pred = model.predict(sample_processed)[0]
    
    if pred == 1:
        st.markdown('<div class="card" style="border-left-color: #dc3545;"><h3 style="color: #dc3545; margin: 0;">High Risk</h3><p style="margin-top: 0.5rem; font-size: 1.1rem;">Bad Credit Assessment</p></div>', unsafe_allow_html=True)
    else:
        st.markdown('<div class="card" style="border-left-color: #28a745;"><h3 style="color: #28a745; margin: 0;">Low Risk</h3><p style="margin-top: 0.5rem; font-size: 1.1rem;">Good Credit Assessment</p></div>', unsafe_allow_html=True)
    
    st.metric("Default Probability", f"{prob:.1%}")

# ==========================================
# 6. SHAP Local Explanation
# ==========================================
with col2:
    st.subheader("Feature Impact Analysis (SHAP)")
    try:
        explainer = shap.TreeExplainer(model)
        bv = explainer.expected_value
        if isinstance(bv, (list, np.ndarray)):
            bv = bv[1] if len(bv) > 1 else bv[0]
        
        sv = explainer.shap_values(sample_processed)
        if isinstance(sv, list):
            sv = sv[1][0]
        else:
            sv = sv[0]
        
        fig, ax = plt.subplots(figsize=(8, 5))
        plt.tight_layout()
        shap.plots.waterfall(shap.Explanation(values=sv, base_values=bv,
                                              data=sample_processed.iloc[0],
                                              feature_names=sample_processed.columns))
        st.pyplot(fig, use_container_width=True)
    except Exception as e:
        st.warning(f"SHAP calculation failed: {e}")

# ==========================================
# 7. DiCE Counterfactual Explanations
# ==========================================
st.markdown("---")
st.subheader("Counterfactual Recommendations")

if pred == 0:
    st.info("Assessment: This applicant is already classified as low risk. No changes recommended.")
else:
    with st.spinner("Analyzing improvement scenarios..."):
        try:
            # Only vary continuous features and simple categorical features
            # Avoid one-hot encoded categorical features
            features_to_vary = ['Credit amount', 'Monthly pay', 'Credit amount_sq']
            
            # Use X_train to build DiCE data
            X_train_with_target = X_train.copy()
            X_train_with_target['Risk'] = 0

            # Ensure binary one-hot columns include both 0 and 1 in the DiCE data
            binary_cols = [c for c in X_train_with_target.columns if set(X_train_with_target[c].dropna().unique()).issubset({0,1})]
            synth_rows = []
            median_vals = X_train_with_target.median(numeric_only=True)
            for col in binary_cols:
                if X_train_with_target[col].nunique() == 1:
                    # create a synthetic row with this binary col = 1 and other numeric columns = median
                    synth = median_vals.to_dict()
                    # set all binary cols to 0
                    for b in binary_cols:
                        synth[b] = 0
                    synth[col] = 1
                    synth_rows.append(synth)
            if synth_rows:
                X_train_with_target = pd.concat([X_train_with_target, pd.DataFrame(synth_rows)], ignore_index=True)

            continuous_features = features_to_vary
            continuous_features = [c for c in continuous_features if c in X_train.columns]

            d = dice_ml.Data(dataframe=X_train_with_target,
                            continuous_features=continuous_features,
                            outcome_name='Risk')
            m = dice_ml.Model(model=wrapped_model, backend="sklearn")
            exp = dice_ml.Dice(d, m, method="random")
            
            # Generate counterfactuals with limited features to vary
            dice_result = exp.generate_counterfactuals(sample_processed, total_CFs=1,
                                                       desired_class="opposite",
                                                       features_to_vary=features_to_vary)
            
            # Display results
            cf_df = dice_result.cf_examples_list[0].final_cfs_df.copy()
            
            # Find changed features
            original_vals = sample_processed.iloc[0]
            changed_features = {}
            for col in features_to_vary:
                if col in X_train.columns:
                    orig = original_vals[col]
                    cf_val = cf_df.iloc[0][col]
                    if abs(orig - cf_val) > 0.001:
                        changed_features[col] = {
                            'Current': round(orig, 2), 
                            'Recommended': round(cf_val, 2),
                            'Change': round(cf_val - orig, 2)
                        }
            
            if changed_features:
                st.write("**Suggested changes to improve credit assessment:**")
                rec_df = pd.DataFrame(changed_features).T
                st.dataframe(rec_df, use_container_width=True)
                st.markdown("""
                **Interpretation:**
                - **Current**: Your current feature value
                - **Recommended**: Suggested value for better assessment
                - **Change**: The difference you need to achieve
                """)
            else:
                st.info("No adjustments needed for key features.")
                
        except Exception as e:
            st.warning(f"Recommendation generation failed: {e}")


