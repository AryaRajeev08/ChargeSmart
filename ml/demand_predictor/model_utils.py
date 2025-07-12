import pandas as pd
from prophet import Prophet

# Load historical data once on startup
df = pd.read_csv("kerala_historical_data.csv")
df['timestamp'] = pd.to_datetime(df['timestamp']).dt.floor('h')

def demand_category(value):
    if value == 0:
        return "Free"
    elif value < 3:
        return "Available"
    elif value < 6:
        return "Busy"
    else:
        return "Very Busy"

def predict_demand(station_id, timestamp_str):
    try:
        timestamp = pd.to_datetime(timestamp_str).floor('h')
        station_df = df[df['station_id'] == station_id].copy()
        if station_df.empty:
            return None, "Station not found"

        # Prepare data
        station_df = station_df[['timestamp', 'bookings_count']]
        station_df = station_df.groupby('timestamp').sum().reset_index()
        station_df.rename(columns={'timestamp': 'ds', 'bookings_count': 'y'}, inplace=True)

        # Fit model
        model = Prophet(daily_seasonality=True)
        model.fit(station_df)

        future = model.make_future_dataframe(periods=48, freq='h')
        forecast = model.predict(future)
        forecast['ds'] = pd.to_datetime(forecast['ds']).dt.floor('h')

        prediction_row = forecast[forecast['ds'] == timestamp]
        if prediction_row.empty:
            return None, "Timestamp not in forecast range"

        yhat = max(0, prediction_row['yhat'].iloc[0])
        category = demand_category(yhat)
        return {
            "station_id": station_id,
            "timestamp": timestamp.isoformat(),
            "predicted_bookings": round(yhat, 2),
            "category": category
        }, None
    except Exception as e:
        return None, str(e)
