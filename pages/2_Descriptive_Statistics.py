# pages/2_Descriptive_Statistics.py
import streamlit as st
import pandas as pd
from utils import load_data

st.title("Descriptive Statistics")

# Load data
df = load_data()

# Descriptive statistics
st.subheader("Descriptive Statistics for 'Valoare'")
st.write(df['Valoare'].describe())

# 1. Total students by education level
st.subheader("Total Students by Education Level")
total_by_education = df.groupby('Nivel_Educatie')['Valoare'].sum().sort_values(ascending=False)
st.write(total_by_education)

# 2. Total students by region (2023)
st.subheader("Total Students by Region (2023)")
total_by_region_2023 = df[df['An'] == 2023].groupby('Regiune')['Valoare'].sum().sort_values(ascending=False)
st.write(total_by_region_2023)

# 3. Total students by teaching language (2023)
st.subheader("Total Students by Teaching Language (2023)")
total_by_language_2023 = df[df['An'] == 2023].groupby('Limba_Predare')['Valoare'].sum().sort_values(ascending=False)
st.write(total_by_language_2023)
