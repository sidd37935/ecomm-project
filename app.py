import streamlit as st
import pandas as pd
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt
from sklearn.linear_model import LinearRegression
from sklearn.model_selection import train_test_split

# 1. Page Config
st.set_page_config(page_title="E-Commerce Analytics", layout="wide")
st.title("📊 AI-Powered E-Commerce Dashboard")

# 2. Load Data (For simplicity, we'll use a CSV, but you can link MySQL here)
@st.cache_data
def load_data():
    df = pd.read_csv("data.csv")
    df['order_date'] = pd.to_datetime(df['order_date'])
    return df

df = load_data()

# 3. Sidebar - KPI Calculations
st.sidebar.header("Filter & Stats")
total_revenue = (df['price'] * df['quantity']).sum()
st.sidebar.metric("Total Revenue", f"${total_revenue:,.2f}")

# 4. Data Visualization (Pandas + Seaborn)
st.subheader("Sales Trends & Category Analysis")
col1, col2 = st.columns(2)

with col1:
    fig, ax = plt.subplots()
    sns.barplot(data=df, x='category', y='price', ax=ax, palette="viridis")
    plt.xticks(rotation=45)
    st.pyplot(fig)

with col2:
    # Correlation Heatmap (NumPy/Pandas)
    fig, ax = plt.subplots()
    sns.heatmap(df.corr(numeric_only=True), annot=True, cmap="coolwarm", ax=ax)
    st.pyplot(fig)

# 5. Machine Learning (Linear Regression)
st.subheader("🚀 Predictive Sales Model")
st.write("Predicting future quantity demand based on price.")

# Preprocessing
X = df[['price']] # Feature
y = df['quantity'] # Target
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2)

model = LinearRegression()
model.fit(X_train, y_train)

# User Input for Prediction
user_price = st.slider("Select Product Price ($)", 1, 1000, 50)
prediction = model.predict([[user_price]])
st.success(f"Predicted Quantity Demand: {int(prediction[0])} units")