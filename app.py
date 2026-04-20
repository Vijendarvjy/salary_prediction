import streamlit as st
import pandas as pd
import joblib
from sklearn.preprocessing import LabelEncoder

# --- Load Model and Encoders ---
try:
    # Load the best model (Random Forest Regressor)
    best_model = joblib.load('random_forest.pkl')
except FileNotFoundError:
    st.error("Model file 'random_forest.pkl' not found. Please ensure the model is saved.")
    st.stop()

# Load original data to recreate LabelEncoder mappings
try:
    original_df = pd.read_csv('Salary_Data.csv')
except FileNotFoundError:
    st.error("Original data file 'Salary_Data.csv' not found. Cannot create label encoder mappings.")
    st.stop()

# Initialize LabelEncoders and fit them on unique non-null values from original data
# This ensures consistency with the training data encoding
gender_encoder = LabelEncoder()
education_encoder = LabelEncoder()
job_title_encoder = LabelEncoder()

# Fit encoders with all unique string values (dropping NaNs first)
gender_encoder.fit(original_df['Gender'].dropna().astype(str).unique())
educoder_classes = original_df['Education Level'].dropna().astype(str).unique()
education_encoder.fit(educoder_classes)
job_encoder_classes = original_df['Job Title'].dropna().astype(str).unique()
job_title_encoder.fit(job_encoder_classes)

# --- Streamlit UI ---
st.set_page_config(page_title="Salary Prediction Dashboard", layout="centered")

st.title("💰 Salary Prediction App")
st.markdown("Enter employee details to predict their salary.")

# Input fields in a sidebar for better organization
st.sidebar.header("Employee Details")

age = st.sidebar.slider("Age", min_value=18, max_value=70, value=30)
gender_selected = st.sidebar.selectbox("Gender", options=list(gender_encoder.classes_))
education_selected = st.sidebar.selectbox("Education Level", options=list(education_encoder.classes_))

# For Job Title, using a selectbox might be long, but it's consistent with other categorical inputs.
# If the list is extremely long, a text input with fuzzy matching or a filtered list might be better.
job_title_selected = st.sidebar.selectbox("Job Title", options=list(job_title_encoder.classes_))

years_experience = st.sidebar.slider("Years of Experience", min_value=0.0, max_value=40.0, value=5.0, step=0.5)

# --- Prediction Logic ---
if st.sidebar.button("Predict Salary"):
    # Encode categorical inputs using the fitted encoders
    gender_encoded = gender_encoder.transform([gender_selected])[0]
    education_encoded = education_encoder.transform([education_selected])[0]
    job_title_encoded = job_title_encoder.transform([job_title_selected])[0]

    # Create a DataFrame for prediction, ensuring column order matches training data
    input_data = pd.DataFrame([[
        age,
        gender_encoded,
        education_encoded,
        job_title_encoded,
        years_experience
    ]],
    columns=['Age', 'Gender', 'Education Level', 'Job Title', 'Years of Experience'])

    # Make prediction
    predicted_salary = best_model.predict(input_data)[0]

    st.subheader("Predicted Salary")

    # Determine color based on salary range for dashboard visualization
    if predicted_salary < 60000:
        color_code = "#FFDDC1"  # Light Peach (low salary)
        color_text = "#E67E22" # Darker Orange
    elif predicted_salary < 120000:
        color_code = "#D4EDDA"  # Light Green (medium salary)
        color_text = "#28A745" # Darker Green
    else:
        color_code = "#BEE9E6"  # Light Teal (high salary)
        color_text = "#17A2B8" # Darker Teal

    # Display predicted salary with color-coded background
    st.markdown(f"""
    <div style='background-color: {color_code}; padding: 20px; border-radius: 10px; text-align: center; border: 2px solid {color_text};'>
        <p style='font-size: 1.2em; color: {color_text}; margin-bottom: 5px;'>Estimated Annual Salary:</p>
        <h2 style='font-size: 2.5em; color: {color_text}; margin-top: 0;'>INR{predicted_salary:,.2f}</h2>
    </div>
    """, unsafe_allow_html=True)

    st.markdown(" ") # Add some space
    st.write(f"*This prediction is based on the Random Forest Regressor model and the inputs provided.*")

# Optional: Add an explanation of features or model info
st.markdown("---")
st.info("This application uses a Random Forest Regressor model trained on historical salary data to make predictions.")
