# 📦 Streamlit App for Predicting the Price of Renting an Apartment in Kyiv

## [Demo App](https://rent-price.streamlit.app/)

[![Streamlit App](https://static.streamlit.io/badges/streamlit_badge_black_white.svg)](https://rent-price.streamlit.app/)

---

## Project Description

This project provides a machine learning-powered web application to predict rental prices for apartments in Kyiv. The app uses a LightGBM regression model with conformal prediction intervals (via MAPIE) and a user-friendly Streamlit interface.

### Features

- Predicts apartment rental prices based on:
  - Location (district, address)
  - Apartment size (full, living, kitchen area)
  - Number of rooms, floor, and building details
  - Amenities and repair state
- Provides prediction intervals for price estimates
- Explains model predictions with feature importance

### Workflow

1. **Data Collection & Cleaning**
    - Scraped rental listings from site
    - Cleaned and preprocessed data
2. **Feature Engineering**
    - Created additional features (e.g., geocoding, area ratios)
    - See [noteboks/real_estate_feature_eng.ipynb](noteboks/real_estate_feature_eng.ipynb)
3. **Model Training**
    - Trained LightGBM model with pipeline ([src/model.py](src/model.py))
    - Saved models to [data/](data/)
4. **Prediction & Explanation**
    - Streamlit app ([app.py](app.py)) loads models and provides predictions with explanations

### Technologies Used

- Python, Pandas, NumPy
- Scikit-learn, LightGBM, XGBoost, MAPIE
- Streamlit (web app)
- Geopy (geocoding)
- Matplotlib (visualization)

---

## Project Structure

```
.
├── app.py                  # Streamlit app entry point
├── src/
│   ├── model.py            # Model training and saving
│   ├── predict_price.py    # Prediction and explanation logic
│   └── utils.py            # Feature engineering utilities
├── data/
│   ├── real_estate_last.csv
│   ├── lgb_model.sav
│   ├── mapie_reg_lgb.sav
│   └── ... (other data and models)
├── noteboks/               # Jupyter notebooks for EDA and feature engineering
├── requirements.txt
└── README.md
```

---

## How to Run Locally

1. **Install dependencies**
    ```sh
    pip install -r requirements.txt
    ```

2. **Train or update the model (optional)**
    ```sh
    python src/model.py
    ```

3. **Start the Streamlit app**
    ```sh
    streamlit run app.py
    ```

---

## Usage

- Fill in apartment details in the sidebar.
- Click "Оцінити" to get a price estimate and prediction interval.
- View feature importance for the prediction.
