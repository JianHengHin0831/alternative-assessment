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
    scaler = joblib.load('scaler.pkl')
    label_encoders = joblib.load('label_encoders.pkl')
    return model, X_train, scaler, label_encoders

model, X_train, scaler, label_encoders = load_all()

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
    df_raw = pd.DataFrame([raw_dict])
    
    exclude_cols = ["Credit amount", "Age", "Duration"]
    categorical_cols = [c for c in df_raw.columns if c not in exclude_cols]
    
    df_encoded = df_raw.copy()
    for col in categorical_cols:
        if col in label_encoders:
            try:
                df_encoded[col] = label_encoders[col].transform(df_raw[[col]])
            except ValueError as e:
                known_classes = label_encoders[col].classes_
                unknown_val = df_raw[col].iloc[0]
                
                print(f"[Warning] Unknown value '{unknown_val}' for column '{col}'")
                print(f"         Known classes: {list(known_classes)}")
                print(f"         Using first known class: '{known_classes[0]}'")
                
                df_encoded[col] = label_encoders[col].transform([known_classes[0]])[0]
    
    scale_cols = ["Credit amount", "Age", "Duration"]
    valid_scale_cols = [c for c in scale_cols if c in df_encoded.columns]
    if valid_scale_cols:
        df_encoded[valid_scale_cols] = scaler.transform(df_encoded[valid_scale_cols])
    
    for col in X_train.columns:
        if col not in df_encoded.columns:
            df_encoded[col] = 0
    
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

# Mapping UI options to training data values
savings_mapping = {
    "little": "little",
    "moderate": "moderate",
    "quite rich": "quite rich",
    "rich": "rich",
    "NA": "NA"
}

# Organize inputs in columns
col1, col2 = st.sidebar.columns(2)

with col1:
    age = st.number_input("Age", min_value=18, max_value=100, value=35, key="age")
    credit_amount = st.number_input("Credit Amount (EUR)", min_value=100, max_value=20000, value=3000, key="credit_amount")
    job = st.selectbox("Job Level", options=[0, 1, 2, 3], 
                       format_func=lambda x: ["Unskilled", "Resident", "Skilled", "Highly Skilled"][x], key="job")
    savings = st.selectbox("Saving Accounts", 
                          options=["little", "moderate", "quite rich", "rich", "NA"], key="savings")

with col2:
    duration = st.number_input("Duration (months)", min_value=1, max_value=72, value=12, key="duration")
    sex = st.selectbox("Sex", options=["female", "male"], key="sex")
    housing = st.selectbox("Housing Type", options=["free", "own", "rent"], key="housing")
    status = st.selectbox("Checking Account", 
                         options=["little", "moderate", "rich", "NA"], key="status")

st.sidebar.markdown("---")

# Purpose in full width
purpose = st.sidebar.selectbox("Purpose", 
                      options=["business", "car", "domestic appliances", "education", 
                              "furniture/equipment", "radio/TV", "repairs", "vacation/others"], 
                      key="purpose")

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

# Show warning if any unknown values were encountered
if hasattr(sample_processed, '_unknown_mappings') and sample_processed._unknown_mappings:
    st.warning(f"Some input values were not found in training data and were mapped to defaults")

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
        st.pyplot(fig, width='stretch')
    except Exception as e:
        st.warning(f"SHAP calculation failed: {e}")

# ==========================================
# 7. DiCE Counterfactual Explanations
# ==========================================
st.markdown("---")
st.subheader("Counterfactual Recommendations")
st.markdown("""
The system analyzes how minimal changes to financial features could alter the risk assessment from **High Risk** to **Low Risk**.
""")

if pred == 0:
    st.info("Assessment: This applicant is already classified as Low Risk. No changes are required.")
else:
    with st.spinner("Generating actionable recommendations (DiCE)..."):
        try:
            immutable_features = ['Age', 'Sex']
            immutable_features = [col for col in immutable_features if col in X_train.columns]
            
            all_features = [col for col in X_train.columns if col != 'Risk']
            features_to_vary = [col for col in all_features if col not in immutable_features]
            
            continuous_features = ['Credit amount']
            continuous_features = [c for c in continuous_features if c in X_train.columns]

            X_train_dice = X_train.copy()
            X_train_dice['Risk'] = model.predict(X_train)

            d = dice_ml.Data(dataframe=X_train_dice,
                             continuous_features=continuous_features,
                             outcome_name='Risk')

            m = dice_ml.Model(model=wrapped_model, backend="sklearn")
            exp = dice_ml.Dice(d, m, method="random")
            dice_result = exp.generate_counterfactuals(
                sample_processed, 
                total_CFs=1, 
                desired_class="opposite",
                features_to_vary=features_to_vary
            )

            cf_df = dice_result.cf_examples_list[0].final_cfs_df.copy()
            original_vals = sample_processed.iloc[0]
            
            changes_found = False
            results_data = []
            
            try:
                scaled_features = list(scaler.get_feature_names_out()) if hasattr(scaler, 'get_feature_names_out') else ["Credit amount", "Age", "Duration"]
            except:
                scaled_features = ["Credit amount", "Age", "Duration"]

            for col in features_to_vary:
                if col in original_vals.index:
                    orig_val = original_vals[col]
                    new_val = cf_df.iloc[0][col]
                    
                    if isinstance(orig_val, (bool, np.bool_)) or isinstance(new_val, (bool, np.bool_)):
                        changed = bool(orig_val) ^ bool(new_val)
                        if changed:
                            changes_found = True
                            results_data.append({
                                "Feature": col,
                                "Current": bool(orig_val),
                                "Recommended": bool(new_val),
                                "Change": "Toggle"
                            })
                    else:
                        diff = new_val - orig_val
                        if abs(diff) > 0.0001:
                            changes_found = True
                            direction = "Decrease" if diff < 0 else "Increase"
                            
                            if col in scaled_features:
                                dummy_scaled = np.zeros((1, len(scaled_features)))
                                col_idx = scaled_features.index(col)
                                
                                dummy_scaled[0, col_idx] = orig_val
                                orig_unscaled = scaler.inverse_transform(dummy_scaled)[0, col_idx]
                                
                                dummy_scaled[0, col_idx] = new_val
                                new_unscaled = scaler.inverse_transform(dummy_scaled)[0, col_idx]
                                
                                results_data.append({
                                    "Feature": col,
                                    "Current": f"{orig_unscaled:.2f}",
                                    "Recommended": f"{new_unscaled:.2f}",
                                    "Suggestion": direction
                                })
                            else:
                                if col in label_encoders:
                                    try:
                                        orig_class_idx = int(round(orig_val))
                                        new_class_idx = int(round(new_val))
                                        
                                        orig_class_name = label_encoders[col].inverse_transform([orig_class_idx])[0]
                                        new_class_name = label_encoders[col].inverse_transform([new_class_idx])[0]
                                        
                                        results_data.append({
                                            "Feature": col,
                                            "Current": orig_class_name,
                                            "Recommended": new_class_name,
                                            "Suggestion": "Change"
                                        })
                                    except:
                                        results_data.append({
                                            "Feature": col,
                                            "Current": f"{orig_val:.4f}",
                                            "Recommended": f"{new_val:.4f}",
                                            "Change": f"{direction} {abs(diff):.4f}"
                                        })
                                else:
                                    results_data.append({
                                        "Feature": col,
                                        "Current": f"{orig_val:.4f}",
                                        "Recommended": f"{new_val:.4f}",
                                        "Change": f"{direction} {abs(diff):.4f}"
                                    })

            if changes_found:
                st.success("Recommendations to Improve Credit Assessment")
                st.dataframe(pd.DataFrame(results_data), width='stretch')
                st.markdown("""
                **Explanation:**
                - **Current**: Your current value
                - **Recommended**: Suggested value
                - **Suggestion**: Direction of adjustment (Increase or Decrease)
                - **Age and Sex**: Remain unchanged (immutable features)
                - **Other Features**: Can be adjusted according to recommendations to improve credit score
                """)
            else:
                st.info("No adjustments found to improve the assessment with current constraints.")

        except Exception as e:
            st.error("Could not generate counterfactuals.")
            st.warning(f"Technical Detail: {str(e)}")