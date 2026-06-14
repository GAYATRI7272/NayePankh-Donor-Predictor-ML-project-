"""Streamlit web app for NayePankh Donor Prediction."""

import os
import pickle
import numpy as np
import pandas as pd
import streamlit as st
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split

st.set_page_config(page_title="NayePankh Donor Predictor", layout="centered")

# Title aur description
st.title("🎯 NayePankh Donor Prediction")
st.markdown("""
Ye web app batata hai ki kaunsa website visitor NayePankh Foundation ke liye donate karega!
Apne details enter karo aur prediction dekho.
""")

st.divider()

# Sidebar me inputs
st.sidebar.header("📊 Visitor Details")

col1, col2 = st.columns(2)

with col1:
    pages_visited = st.slider("Pages Visited", 1, 15, 7, help="Kitne pages dekhe?")
    mobile_user = st.selectbox("Mobile User?", ["No (Desktop)", "Yes (Mobile)"])

with col2:
    time_on_site = st.slider("Time on Site (seconds)", 30, 600, 300, help="Site pe kitna time spend kiya?")
    viewed_campaign = st.selectbox("Viewed Campaign?", ["No", "Yes"])

city_kanpur = st.selectbox("From Kanpur?", ["No", "Yes"])
from_instagram = st.selectbox("From Instagram?", ["No", "Yes"])

# Convert to binary
mobile_user = 1 if mobile_user == "Yes (Mobile)" else 0
viewed_campaign = 1 if viewed_campaign == "Yes" else 0
city_kanpur = 1 if city_kanpur == "Yes" else 0
from_instagram = 1 if from_instagram == "Yes" else 0

# Train model on the fly (we can also load a saved model)
@st.cache_resource
def train_model():
    df = pd.read_csv(os.path.join("data", "donor_data.csv"))
    X = df.drop("donated", axis=1)
    y = df["donated"]
    
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42, stratify=y)
    
    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train)
    X_test_scaled = scaler.transform(X_test)
    
    model = LogisticRegression(random_state=42, max_iter=1000)
    model.fit(X_train_scaled, y_train)
    
    return model, scaler, X.columns

# Get model
model, scaler, feature_names = train_model()

# Create input array
input_data = np.array([[pages_visited, time_on_site, mobile_user, viewed_campaign, city_kanpur, from_instagram]])
input_scaled = scaler.transform(input_data)

# Make prediction
prediction = model.predict(input_scaled)[0]
probability = model.predict_proba(input_scaled)[0][1]

# Display results
st.divider()
st.subheader("🔮 Prediction Result")

col1, col2 = st.columns(2)

with col1:
    if prediction == 1:
        st.success(f"**Will DONATE!** 💰")
    else:
        st.error(f"**Unlikely to donate** 🚫")

with col2:
    st.metric("Probability", f"{probability*100:.1f}%")

# Feature impact explanation
st.divider()
st.subheader("📈 What Affects Donation?")

feature_importance = {
    "Time on Site": "Most important! Jyada time = Jyada donation chance",
    "Pages Visited": "Zyada explore = Interested",
    "Viewed Campaign": "Campaign dekha toh +15% chance",
    "From Instagram": "+8% chance agar Instagram se aye",
    "Mobile User": "-5% chance agar mobile se aye",
    "From Kanpur": "+1% chance",
}

for feature, description in feature_importance.items():
    st.write(f"• **{feature}**: {description}")

# Footer
st.divider()
st.markdown("""
---
*NayePankh ML Project | Developed for donor conversion prediction*
""")
