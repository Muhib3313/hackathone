import pandas as pd
import streamlit as st
from sklearn.ensemble import RandomForestClassifier
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline

# Load and preprocess data
df = pd.read_csv('cleaned_data.csv')

# Feature engineering
df['launch_date'] = pd.to_datetime(df['launch_date'])
df['launch_year'] = df['launch_date'].dt.year
df['launch_month'] = df['launch_date'].dt.month
df['launch_day'] = df['launch_date'].dt.day
df['launch_day_of_week'] = df['launch_date'].dt.dayofweek
df['success'] = df['success'].astype(int)

# Prepare features and target
X = df.drop(['mission_name', 'success', 'location', 'launch_date'], axis=1)
y = df['success']

# Preprocessing
numeric_features = ['payload_mass', 'launch_year', 'launch_month', 'launch_day', 'launch_day_of_week']
numeric_transformer = StandardScaler()

categorical_features = ['rocket_name', 'orbit', 'site_name']
categorical_transformer = OneHotEncoder(handle_unknown='ignore')

preprocessor = ColumnTransformer(
    transformers=[
        ('num', numeric_transformer, numeric_features),
        ('cat', categorical_transformer, categorical_features)
    ])

# Pipeline
pipeline = Pipeline([
    ('preprocessor', preprocessor),
    ('classifier', RandomForestClassifier(random_state=42))
])

# Train model
pipeline.fit(X, y)

# --- Streamlit App ---

st.set_page_config(page_title="🚀 SpaceX Launch Success Predictor", layout="centered")

st.title("🚀 SpaceX Launch Success Predictor")
st.write("Enter launch parameters below to predict success probability.")

# User input form
with st.form("launch_form"):
    rocket_name = st.selectbox("Rocket name", df['rocket_name'].unique())
    payload_mass = st.number_input("Payload mass (kg)", min_value=0.0, value=5000.0)
    orbit = st.selectbox("Orbit", df['orbit'].unique())
    site_name = st.selectbox("Launch site", df['site_name'].unique())
    launch_date = st.date_input("Launch date")

    submitted = st.form_submit_button("Predict")

if submitted:
    launch_date = pd.to_datetime(launch_date)
    input_df = pd.DataFrame({
        'rocket_name': [rocket_name],
        'payload_mass': [payload_mass],
        'orbit': [orbit],
        'site_name': [site_name],
        'launch_year': [launch_date.year],
        'launch_month': [launch_date.month],
        'launch_day': [launch_date.day],
        'launch_day_of_week': [launch_date.dayofweek]
    })

    prediction = pipeline.predict(input_df)[0]
    probability = pipeline.predict_proba(input_df)[0]

    st.subheader("Prediction Results")
    if prediction:
        st.success(f"✅ Predicted: Successful Launch with {probability[1]*100:.1f}% confidence")
    else:
        st.error(f"❌ Predicted: Launch Failure with {probability[0]*100:.1f}% confidence")

    st.subheader("Detailed Probabilities")
    st.progress(probability[1])
    st.text(f"Success Probability: {probability[1]*100:.2f}%")
    st.text(f"Failure Probability: {probability[0]*100:.2f}%")
