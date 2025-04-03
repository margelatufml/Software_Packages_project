# utils.py
import pandas as pd

def load_data():
    file_path = "setDate.xlsx"  # Update path if needed
    df = pd.read_excel(
        file_path,
        sheet_name="Sheet 1 - exportPivot_SCL103B",
        skiprows=2,   # Skip metadata rows
        usecols="A:F",  # Select relevant columns
        header=0      # Use the third row as header
    )

    # Rename columns for clarity
    df.columns = [
        'Nivel_Educatie',  # Education Level
        'Limba_Predare',   # Teaching Language
        'Regiune',         # Region
        'Perioada',        # Period
        'UM',              # Unit of Measurement
        'Valoare'          # Value (Number of People)
    ]

    # Extract year from 'Perioada' column
    df['An'] = df['Perioada'].str.extract(r'(\d{4})').astype(int)

    # Clean region names (remove "Regiunea " and extra spaces)
    df['Regiune'] = df['Regiune'].str.replace('Regiunea ', '', regex=False).str.strip()

    # Ensure 'Valoare' is numeric; coerce errors to NaN
    df['Valoare'] = pd.to_numeric(df['Valoare'], errors='coerce')

    # Drop rows where 'Valoare' is NaN
    df = df.dropna(subset=['Valoare'])

    return df
