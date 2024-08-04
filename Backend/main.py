import pandas as pd
import numpy as np
from statsmodels.tsa.holtwinters import ExponentialSmoothing
from prophet import Prophet
from sklearn.ensemble import GradientBoostingRegressor
from sklearn.model_selection import train_test_split
from datetime import datetime


# Load and preprocess data
df = pd.read_csv('dci_scores.csv')
df['Date'] = pd.to_datetime(df['Date'])
df['DayOfYear'] = df['Date'].dt.dayofyear
df['Year'] = df['Date'].dt.year

# Separate data by year and corps
def prepare_data(df, corps):
    corps_data = df[df['Corps Name'] == corps].sort_values('Date')
    return corps_data


# Function to get previous scores within the same year
def previous_scores(date, corps):
    date_object = pd.to_datetime(date)
    year = date_object.year
    previous_scores = df[(df['Corps Name'] == corps)]
    return previous_scores

# Train models for each corps
def train_models(corps_data):
    # Exponential Smoothing
    es_data = corps_data.set_index('Date')['Score']
    es_model = ExponentialSmoothing(es_data, trend='add', seasonal=None)
    es_results = es_model.fit()

    # Prophet
    prophet_data = corps_data[['Date', 'Score']].rename(columns={'Date': 'ds', 'Score': 'y'})
    prophet_model = Prophet()
    prophet_model.fit(prophet_data)

    # Gradient Boosting
    X = corps_data[['DayOfYear', 'Year']]
    y = corps_data['Score']
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
    gb_model = GradientBoostingRegressor()
    gb_model.fit(X_train, y_train)

    return es_results, prophet_model, gb_model


def predict_score(date, corps, models):
    es, prophet, gb = models
    date = pd.to_datetime(date)
    
    # Check if date exists in CSV
    existing_score = df[(df['Date'] == date) & (df['Corps Name'] == corps)]['Score']
    if not existing_score.empty:
        return existing_score.iloc[0]

    # If date is between September and June, return None
    if date.month in range(9, 13) or date.month in range(1, 7):
        return None

    # Make predictions
    es_pred = es.forecast(1).iloc[0]
    prophet_pred = prophet.predict(pd.DataFrame({'ds': [date]}))['yhat'].iloc[0]
    gb_pred = gb.predict([[date.dayofyear, date.year]])[0]

    # Combine predictions with more weight on GB and Prophet
    combined_pred = (es_pred * 0.2 + prophet_pred * 0.4 + gb_pred * 0.4)

    # Get previous scores
    previous_scores = df[(df['Date'] < date) & (df['Corps Name'] == corps)]['Score']
    
    if not previous_scores.empty:
        last_score = previous_scores.iloc[-1]
        last_date = df[(df['Date'] < date) & (df['Corps Name'] == corps)]['Date'].iloc[-1]
        
        # Calculate days since last performance
        days_since_last = (date - last_date).days
        
        # Calculate expected improvement based on days since last performance
        expected_improvement = days_since_last * 0.05  # Adjust this factor as needed
        
        # Ensure prediction is not lower than previous score plus expected improvement
        combined_pred = max(combined_pred, last_score + expected_improvement)
        
    
    # Ensure the score doesn't exceed 100
    combined_pred = min(combined_pred, 100)

    return round(combined_pred, 3)

# Train models for each corps
corps_list = df['Corps Name'].unique()
models = {corps: train_models(prepare_data(df, corps)) for corps in corps_list}


if __name__ == "__main__":
    num_predictions = input("Enter the number of predictions you want to run: ")
    try:
        num_predictions = int(num_predictions)
    except ValueError:
        print("Please enter a valid number.")
        exit()

    predictions_input = []
    for _ in range(num_predictions):
        date_input = input("Enter date (YYYY-MM-DD): ")
        try:
            # Attempt to convert the input string to a datetime object
            date_object = datetime.strptime(date_input, '%Y-%m-%d')
            # Check if the date is within the specified ranges
            if date_object.month in range(9, 13) or date_object.month in range(1, 7):
                print("Predictions are not available for dates between September and June. Please enter a different date.")
                continue
        except ValueError:
            # If an exception is caught, inform the user and continue to the next iteration of the loop
            print("Invalid date format. Please enter a date in the format YYYY-MM-DD.")
            continue

        corps_input = input("Enter corps name: ")
        predictions_input.append((date_object, corps_input))

    for date_object, corps_input in predictions_input:
        if corps_input in models:
            predicted_score = predict_score(date_object, corps_input, models[corps_input])
            print(f"Predicted score for {corps_input} on {date_object.strftime('%Y-%m-%d')}: {predicted_score}")
        else:
            print(f"Corps name '{corps_input}' not found. Please enter a valid corps name.")