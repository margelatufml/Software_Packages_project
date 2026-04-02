# SAS Data Analysis Platform

A Streamlit-based analytics platform for processing and analyzing educational datasets using SAS integration and Python.

## Tech Stack

- **Frontend:** Streamlit (interactive web UI)
- **Analytics:** SAS integration via SASPy
- **Data Processing:** Pandas, Python
- **File Handling:** DOCX text extraction, Excel processing

## Features

- Interactive data analysis with Streamlit UI
- SAS script execution and integration
- Educational dataset analysis (sample data included)
- Document text extraction (DOCX, TXT)
- Excel data processing and visualization

## Project Structure

```
├── app.py                    # Main Streamlit application
├── pages/                    # Multi-page Streamlit app
├── utils.py                  # Utility functions
├── SAS/                      # SAS scripts and analysis
├── requirements.txt          # Python dependencies
└── sascfg_personal.py.sample # SAS configuration template
```

## Getting Started

```bash
pip install -r requirements.txt
streamlit run app.py
```

