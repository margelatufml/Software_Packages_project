# pages/1_Data_Overview.py
import streamlit as st
import pandas as pd
from utils import load_data

def app():
    pass

####################
# Streamlit Section
####################
st.title("Data Overview")

# Load data
df = load_data()

st.write("On this page, you can explore the dataset’s structure, columns, and sample rows.")

# Show data types
st.subheader("Column Names and Data Types")
st.write(df.dtypes)

# Check missing values
st.subheader("Missing Values")
st.write(df.isnull().sum())

# Display a sample of rows
st.subheader("Data Preview")
num_rows = st.slider("Select number of rows to display:", min_value=1, max_value=50, value=5)
st.dataframe(df.head(num_rows))
