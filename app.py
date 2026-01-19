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
    
    /* Tab styling */
    [data-testid="stTabs"] {
        background-color: transparent;
    }
    
    [role="tablist"] {
        background-color: #f0f2f6;
        border-radius: 8px;
        padding: 4px;
        margin-bottom: 1.5rem;
    }
    
    [role="tab"] {
        padding: 0.75rem 1.5rem !important;
        border-radius: 6px;
        font-weight: 500;
        color: #2c3e50 !important;
        background-color: transparent !important;
        border: none !important;
        transition: all 0.3s ease;
        margin-right: 4px;
    }
    
    [role="tab"]:hover {
        background-color: rgba(31, 119, 180, 0.1) !important;
    }
    
    [role="tab"][aria-selected="true"] {
        background-color: #1f77b4 !important;
        color: white !important;
        box-shadow: 0 2px 8px rgba(31, 119, 180, 0.3);
    }
    
    /* Tab content styling */
    [data-testid="stTabContent"] {
        padding: 2rem 1rem;
        border-radius: 8px;
        background-color: #ffffff;
    }
    
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
    
    # Map 'NA' string values to np.nan for proper encoding
    df_raw = df_raw.replace('NA', np.nan)
    
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
    "rather not to say / no account": "NA"
}

checking_mapping = {
    "little": "little",
    "moderate": "moderate",
    "rich": "rich",
    "rather not to say / no account": "NA"
}

# Job level mapping
job_mapping = {
    0: "Unskilled",
    1: "Resident",
    2: "Skilled",
    3: "Highly Skilled"
}

# Organize inputs in columns
col1, col2 = st.sidebar.columns(2)

with col1:
    age = st.number_input("Age", min_value=19, max_value=75, value=35, key="age")
    credit_amount = st.number_input("Credit Amount (EUR)", min_value=100, max_value=20000, value=3000, key="credit_amount")
    job = st.selectbox("Job Level", options=[0, 1, 2, 3], 
                       format_func=lambda x: ["Unskilled", "Resident", "Skilled", "Highly Skilled"][x], key="job")
    savings = st.selectbox("Saving Accounts", 
                          options=["little", "moderate", "quite rich", "rich", "rather not to say / no account"], key="savings")

with col2:
    duration = st.number_input("Duration (months)", min_value=6, max_value=48, value=12, key="duration")
    sex = st.selectbox("Sex", options=["female", "male"], key="sex")
    housing = st.selectbox("Housing Type", options=["free", "own", "rent"], key="housing")
    status = st.selectbox("Checking Account", 
                         options=["little", "moderate", "rich", "rather not to say / no account"], key="status")

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
    'Saving accounts': savings_mapping.get(savings, savings),
    'Checking account': checking_mapping.get(status, status),
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

col1, col2, col3 = st.columns([1.2, 1.2, 1.3])

with col1:
    st.subheader("Prediction Result")
    prob = model.predict_proba(sample_processed)[0][1]
    pred = model.predict(sample_processed)[0]
    
    if pred == 1:
        st.markdown('<div class="card" style="border-left-color: #dc3545;"><h3 style="color: #dc3545; margin: 0;">High Risk</h3><p style="margin-top: 0.5rem; font-size: 1.1rem;">Bad Credit Assessment</p></div>', unsafe_allow_html=True)
    else:
        st.markdown('<div class="card" style="border-left-color: #28a745;"><h3 style="color: #28a745; margin: 0;">Low Risk</h3><p style="margin-top: 0.5rem; font-size: 1.1rem;">Good Credit Assessment</p></div>', unsafe_allow_html=True)
    
    st.metric("Default Probability", f"{prob:.1%}")

with col2:
    pass

with col3:
    st.subheader("Risk Gauge")

    fig, ax = plt.subplots(figsize=(6, 4), subplot_kw=dict(projection='polar'))

    # Risk score (0–1)
    risk_score = prob

    ax.set_thetamin(0)
    ax.set_thetamax(180)

    theta = np.linspace(np.pi, 0, 200)

    # Risk zones
    low_risk_theta = theta[theta >= np.pi / 2]
    high_risk_theta = theta[theta < np.pi / 2]

    ax.fill_between(low_risk_theta, 0, 1, color="#28a745", alpha=0.3)
    ax.fill_between(high_risk_theta, 0, 1, color="#dc3545", alpha=0.3)

    needle_theta = np.pi * (1 - risk_score)
    ax.plot([needle_theta, needle_theta], [0, 1], color="#1f77b4", linewidth=3)
    ax.plot(needle_theta, 1, "o", color="#1f77b4", markersize=10)

    ax.set_ylim(0, 1.1)
    ax.set_theta_zero_location("E")  
    ax.set_theta_direction(1)        
    ax.set_xticks([np.pi, np.pi/2, 0])
    ax.set_xticklabels(["0%", "50%", "100%"])
    ax.set_yticks([])
    ax.grid(False)

    plt.tight_layout()
    st.pyplot(fig, width="stretch")

    # =========================
    # 4. Risk Label
    # =========================
    if risk_score < 0.5:
        st.markdown(
            "<div style='text-align:center; font-weight:bold; color:#28a745;'>🟢 Low Risk</div>",
            unsafe_allow_html=True
        )
    else:
        st.markdown(
            "<div style='text-align:center; font-weight:bold; color:#dc3545;'>🔴 High Risk</div>",
            unsafe_allow_html=True
        )


# ==========================================
# 7. Multi-Perspective Model Explanations
# ==========================================
st.markdown("---")
st.subheader("Model Explanations")
st.markdown("Choose an explanation perspective suited to your role")

# Create tabs for different explanation methods
tab1, tab2, tab3 = st.tabs(["Global Explanation (Auditor)", "Local Explanation (Loan Officer)", "Counterfactual Recommendations (Applicant)"])

# ==========================================
# TAB 1: Global Explanation
# ==========================================
with tab1:
    st.markdown("### Global Model Behavior Analysis")
    st.markdown("""
    This section provides an overview of how the model makes decisions across all applicants in the dataset.
    """)
    
    # Display SHAP summary images if available
    col1, col2 = st.columns(2)
    
    with col1:
        st.markdown("#### SHAP Summary Plot")
        try:
            from PIL import Image
            img_summary = Image.open('shap-summary.png')
            st.image(img_summary, width='stretch')
        except:
            st.info("SHAP summary visualization not available. Please ensure 'shap-summary.png' exists in the application directory.")
    
    with col2:
        st.markdown("#### SHAP Dependence Heatmap")
        try:
            from PIL import Image
            img_heatmap = Image.open('shap-heatmap.png')
            st.image(img_heatmap, width='stretch')
        except:
            st.info("SHAP heatmap visualization not available. Please ensure 'shap-heatmap.png' exists in the application directory.")
    
    # Display SHAP summary images if available
    col1, col2 = st.columns(2)
    
    with col1:
        st.markdown("#### SHAP Bar Plot")
        try:
            from PIL import Image
            img_summary = Image.open('shap-bar-plot.png')
            st.image(img_summary, width='stretch')
        except:
            st.info("SHAP summary visualization not available. Please ensure 'shap-summary.png' exists in the application directory.")
    
    with col2:
        st.markdown("#### SHAP Dependence Plot")
        try:
            from PIL import Image
            img_heatmap = Image.open('shap-dependence.png')
            st.image(img_heatmap, width='stretch')
        except:
            st.info("SHAP heatmap visualization not available. Please ensure 'shap-heatmap.png' exists in the application directory.")
    

    st.markdown("---")
    st.markdown("### Key Insights")
    
    insights = {
        "Key Drivers": "Checking account, Duration, and Credit amount dominate the model's decisions, reflecting liquidity and loan burden.",
        "Economic Logic": "Lower checking account balances push predictions toward High Risk, consistent with credit risk theory.",
        "Fairness Risk": "Demographic features (Age, Sex) have non-negligible impact and require fairness review for regulatory compliance.",
        "Stability Concern": "Sharp importance drop after top features suggests over-reliance on Checking account, risking robustness if data is missing."
    }
    
    for title, insight in insights.items():
        st.markdown(f"**{title}**")
        st.markdown(f"> {insight}")
        st.markdown("")

# ==========================================
# TAB 2: SHAP Local Explanation
# ==========================================
with tab2:
    st.markdown("### Feature Impact on This Application")
    st.markdown("""
    This waterfall plot shows how each feature pushes the prediction up or down for this specific applicant.
    """)
    
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
        
        fig, ax = plt.subplots(figsize=(10, 6))

        plt.tight_layout()
        shap.plots.waterfall(shap.Explanation(values=sv, base_values=bv,
                                              data=sample_processed.iloc[0],
                                              feature_names=sample_processed.columns))
        st.pyplot(fig, width='stretch')
        
        # Generate simple explanation
        st.markdown("---")
        st.markdown("### Simple Explanation")
        
        positive_features = []
        negative_features = []
        
        for i, feature_name in enumerate(sample_processed.columns):
            if sv[i] > 0.01:  # Small threshold to avoid noise
                positive_features.append(feature_name)
            elif sv[i] < -0.01:
                negative_features.append(feature_name)
        
        # Build explanation text
        if positive_features and negative_features:
            pos_str = ", ".join(positive_features)
            neg_str = ", ".join(negative_features)
            explanation = f"**{pos_str}** are pushing toward **High Risk**, while **{neg_str}** are pushing toward **Low Risk**."
        elif positive_features:
            pos_str = ", ".join(positive_features)
            explanation = f"**{pos_str}** are pushing toward **High Risk**."
        elif negative_features:
            neg_str = ", ".join(negative_features)
            explanation = f"**{neg_str}** are pushing toward **Low Risk**."
        else:
            explanation = "No significant feature impacts detected."
        
        st.markdown(explanation)
        
        st.markdown("""
        **How to read this plot:**
        - **Base value**: Starting prediction (average across all applicants)
        - **Red arrows**: Features pushing toward High Risk
        - **Blue arrows**: Features pushing toward Low Risk
        - **Final prediction**: Result at the top after all features are considered
        """)
    except Exception as e:
        st.warning(f"SHAP calculation failed: {e}")

# ==========================================
# TAB 3: DiCE Counterfactual Explanations
# ==========================================
with tab3:
    st.markdown("### Actionable Recommendations to Improve Assessment")
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
            
                continuous_features = ['Credit amount', 'Duration','Age']
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
                
                # Define continuous scaled features
                continuous_scale_features = ["Credit amount", "Age", "Duration"]

                for col in features_to_vary:
                    if col in original_vals.index:
                        orig_val = original_vals[col]
                        new_val = cf_df.iloc[0][col]
                        
                        # Skip if values are essentially the same
                        if abs(float(orig_val) - float(new_val)) < 0.0001:
                            continue
                        
                        changes_found = True
                        
                        # Handle continuous scaled features (need inverse scaling)
                        if col in continuous_scale_features:
                            try:
                                dummy_scaled = np.zeros((1, len(continuous_scale_features)))
                                col_idx = continuous_scale_features.index(col)
                                
                                dummy_scaled[0, col_idx] = orig_val
                                orig_unscaled = scaler.inverse_transform(dummy_scaled)[0, col_idx]
                                
                                dummy_scaled[0, col_idx] = new_val
                                new_unscaled = scaler.inverse_transform(dummy_scaled)[0, col_idx]
                                
                                direction = "Decrease" if new_unscaled < orig_unscaled else "Increase"
                                results_data.append({
                                    "Feature": col,
                                    "Current": f"{orig_unscaled:.2f}",
                                    "Recommended": f"{new_unscaled:.2f}",
                                    "Suggestion": direction
                                })
                            except Exception as e:
                                results_data.append({
                                    "Feature": col,
                                    "Current": f"{orig_val:.4f}",
                                    "Recommended": f"{new_val:.4f}",
                                    "Suggestion": "Change"
                                })
                        
                        # Handle categorical encoded features (decode back to original class names)
                        elif col in label_encoders:
                            try:
                                orig_class_idx = int(round(float(orig_val)))
                                new_class_idx = int(round(float(new_val)))
                                
                                # Special handling for Job column
                                if col == "Job":
                                    orig_class_name = job_mapping.get(orig_class_idx, str(orig_class_idx))
                                    new_class_name = job_mapping.get(new_class_idx, str(new_class_idx))
                                else:
                                    orig_class_name = label_encoders[col].inverse_transform([orig_class_idx])[0]
                                    new_class_name = label_encoders[col].inverse_transform([new_class_idx])[0]
                                
                                # Replace 'nan' with user-friendly label
                                if str(orig_class_name) == 'nan':
                                    orig_class_name = 'rather not to say / no account'
                                if str(new_class_name) == 'nan':
                                    new_class_name = 'rather not to say / no account'
                                
                                results_data.append({
                                    "Feature": col,
                                    "Current": str(orig_class_name),
                                    "Recommended": str(new_class_name),
                                    "Suggestion": "Change"
                                })
                            except Exception as decode_err:
                                results_data.append({
                                    "Feature": col,
                                    "Current": f"{orig_val:.4f}",
                                    "Recommended": f"{new_val:.4f}",
                                    "Suggestion": "Change"
                                })
                        
                        # Handle other numeric features
                        else:
                            try:
                                # Special handling for Job column
                                if col == "Job":
                                    orig_class_idx = int(round(float(orig_val)))
                                    new_class_idx = int(round(float(new_val)))
                                    
                                    orig_class_name = job_mapping.get(orig_class_idx, str(orig_class_idx))
                                    new_class_name = job_mapping.get(new_class_idx, str(new_class_idx))
                                    
                                    results_data.append({
                                        "Feature": col,
                                        "Current": str(orig_class_name),
                                        "Recommended": str(new_class_name),
                                        "Suggestion": "Change"
                                    })
                                else:
                                    diff = float(new_val) - float(orig_val)
                                    direction = "Decrease" if diff < 0 else "Increase"
                                    results_data.append({
                                        "Feature": col,
                                        "Current": f"{float(orig_val):.4f}",
                                        "Recommended": f"{float(new_val):.4f}",
                                        "Suggestion": direction
                                    })
                            except:
                                results_data.append({
                                    "Feature": col,
                                    "Current": str(orig_val),
                                    "Recommended": str(new_val),
                                    "Suggestion": "Change"
                                })

                if changes_found:
                    st.success("Recommendations to Improve Credit Assessment")
                    
                    # Generate explanation sentence
                    change_parts = []
                    for item in results_data:
                        change_parts.append(f"**{item['Feature']}** from *{item['Current']}* to *{item['Recommended']}*")
                    
                    if len(change_parts) == 1:
                        explanation = f"If you change {change_parts[0]}, then the assessment will change to **Low Risk**."
                    elif len(change_parts) == 2:
                        explanation = f"If you change {change_parts[0]} and {change_parts[1]}, then the assessment will change to **Low Risk**."
                    else:
                        explanation = f"If you change {', '.join(change_parts[:-1])}, and {change_parts[-1]}, then the assessment will change to **Low Risk**."
                    
                    st.info(explanation)
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