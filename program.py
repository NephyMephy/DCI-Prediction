import pandas as pd
import numpy as np
from sklearn.preprocessing import LabelEncoder
from sklearn.metrics import mean_absolute_error, mean_squared_error
from statsmodels.tsa.arima.model import ARIMA
import matplotlib.pyplot as plt
from datetime import datetime

# 1. Data structure and preparation
def load_and_prepare_data(file_path):
    df = pd.read_csv(file_path)
    df['Date'] = pd.to_datetime(df['Date'])
    df = df.sort_values(['Season', 'Date'])
    df = df.dropna()
    
    le = LabelEncoder()
    df['Corps_Encoded'] = le.fit_transform(df['Corps Name'])
    
    return df, le

# 2. Feature engineering
def engineer_features(df):
    df['DayOfWeek'] = df['Date'].dt.dayofweek
    df['WeekOfSeason'] = df.groupby('Season')['Date'].transform(lambda x: ((x - x.min()).dt.days // 7) + 1)
    df['DaysSinceSeasonStart'] = df.groupby('Season')['Date'].transform(lambda x: (x - x.min()).dt.days)
    
    df['AvgLast3'] = df.groupby(['Season', 'Corps Name'])['Score'].rolling(window=3, min_periods=1).mean().reset_index(level=[0,1], drop=True)
    
    def get_season_rank(group):
        return group.sort_values('Score', ascending=False).index.to_series().rank(method='min')
    
    df['SeasonRank'] = df.groupby('Season').apply(get_season_rank).reset_index(level=0, drop=True)
    
    return df

# 3. Model selection and training
def train_arima_models(df):
    models = {}
    for corps in df['Corps Name'].unique():
        corps_data = df[df['Corps Name'] == corps].sort_values('Date')
        model = ARIMA(corps_data['Score'], order=(1,1,1))
        models[corps] = model.fit()
    return models

# 4. Prediction function
def predict_score(models, date, corps_name, df, le):
    date = pd.to_datetime(date)
    season = date.year
    
    if corps_name not in models:
        raise ValueError(f"No model available for {corps_name}")
    
    model = models[corps_name]
    
    corps_data = df[df['Corps Name'] == corps_name].sort_values('Date')
    
    if corps_data.empty:
        raise ValueError(f"No data available for {corps_name}")
    
    # Check if the date exists in the data
    known_score = corps_data[corps_data['Date'] == date]['Score']
    if not known_score.empty:
        return known_score.values[0]
    
    last_known_date = corps_data['Date'].max()
    days_since_last = (date - last_known_date).days
    
    if days_since_last < 0:
        raise ValueError(f"Prediction date is before the last known date for {corps_name}")
    
    forecast = model.forecast(steps=days_since_last+1)
    predicted_score = forecast.iloc[-1]
    
    return predicted_score

# 5. Model evaluation
def evaluate_models(models, df):
    y_true = []
    y_pred = []
    for corps, model in models.items():
        corps_data = df[df['Corps Name'] == corps].sort_values('Date')
        actual = corps_data['Score'].values
        predicted = model.fittedvalues
        y_true.extend(actual[1:])  # Exclude the first value as ARIMA uses it for differencing
        y_pred.extend(predicted[1:])
    
    mae = mean_absolute_error(y_true, y_pred)
    rmse = np.sqrt(mean_squared_error(y_true, y_pred))
    return mae, rmse

# 6. User interface
def user_interface(models, df, le):
    while True:
        date = input("Enter date (YYYY-MM-DD) or 'q' to quit: ")
        if date.lower() == 'q':
            break
        corps_name = input("Enter corps name: ")
        
        try:
            predicted_score = predict_score(models, date, corps_name, df, le)
            print(f"Predicted score for {corps_name} on {date}: {predicted_score:.3f}")
        except Exception as e:
            print(f"Error: {e}")

# Main execution
if __name__ == "__main__":
    # Load and prepare data
    df, le = load_and_prepare_data('dci_scores.csv')
    
    # Engineer features
    df = engineer_features(df)
    
    # Train models
    models = train_arima_models(df)
    
    # Evaluate models
    mae, rmse = evaluate_models(models, df)
    print(f"Model evaluation: MAE = {mae:.3f}, RMSE = {rmse:.3f}")
    
    # Plot actual vs predicted scores for a sample corps
    sample_corps = df['Corps Name'].iloc[0]
    sample_data = df[df['Corps Name'] == sample_corps].sort_values('Date')
    sample_model = models[sample_corps]
    
    plt.figure(figsize=(12, 6))
    plt.plot(sample_data['Date'], sample_data['Score'], label='Actual')
    plt.plot(sample_data['Date'], sample_model.fittedvalues, label='Predicted')
    plt.title(f'Actual vs Predicted DCI Scores for {sample_corps}')
    plt.xlabel('Date')
    plt.ylabel('Score')
    plt.legend()
    plt.show()
    
    # Start user interface
    user_interface(models, df, le)