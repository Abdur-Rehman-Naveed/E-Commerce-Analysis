# E-Commerce Analysis

## Overview
This project is a comprehensive data analysis and predictive modeling application for retail e-commerce data. It features a Streamlit web application (`app.py`) for interactive data exploration and a comprehensive analysis script (`retail_analysis.py`) for generating detailed reports, probability distributions, customer segmentation (RFM & K-Means), and sales predictions.

## Features
- **Descriptive Statistics**: Core distributions and order-level statistics.
- **Probability & Distributions**: Repeat purchase likelihood and normal/Poisson fits.
- **Confidence Intervals**: 95% & 99% CIs for key business metrics.
- **Regression & Prediction**: Predictive modeling for customer spend and purchase frequency.
- **Customer Segmentation**: K-Means clustering combined with RFM analysis.
- **Time Series & Seasonality**: Monthly revenue trends and seasonality indices.
- **Interactive Dashboard**: Built with Streamlit for a user-friendly data intelligence interface.

## Prerequisites
- Python 3.8 or higher.
- Ensure the `online_retail_II.xlsx` dataset is placed in the root directory.

## Setup Instructions

1. **Clone the repository** (if applicable) or download the source code:
   ```bash
   git clone <repository-url>
   cd "E-Commerce Analysis"
   ```

2. **Create a virtual environment (optional but recommended)**:
   ```bash
   python -m venv venv
   # On Windows
   venv\Scripts\activate
   # On macOS/Linux
   source venv/bin/activate
   ```

3. **Install the required libraries**:
   ```bash
   pip install -r requirements.txt
   ```

## Running the Project

### 1. Interactive Streamlit App
To launch the interactive web dashboard, run:
```bash
streamlit run app.py
```

### 2. Standalone Analysis Script
To run the full analysis pipeline and generate charts in the `outputs/` folder, run:
```bash
python retail_analysis.py
```
*(Make sure the `outputs/` folder exists or update the `FIGURE_SAVE_DIR` path in the script if necessary)*
