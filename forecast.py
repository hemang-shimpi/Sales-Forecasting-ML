import pandas as pd
from prophet import Prophet
from prophet.plot import plot_plotly
from sklearn.metrics import mean_absolute_error, mean_squared_error
import numpy as np

def load_data(path):
    data = pd.read_csv(path)

    data['sales'] = data[['NA_Sales', 'EU_Sales', 'JP_Sales', "Other_Sales", "Global_Sales"]].sum(axis=1, skipna=True)

    data = data.dropna(subset=['Year'])

    data['ds'] = pd.to_datetime(data['Year'].astype(int).astype(str) + '-01-01')
    data['y'] = data['sales']

    df = data[['ds', 'y']]

    return df


def main():

    df = load_data("vgsales.csv")

    model = Prophet(yearly_seasonality=False)
    model.fit(df)

    future = model.make_future_dataframe(periods=5, freq='Y')

    forecast = model.predict(future)

    forecast[['ds', 'yhat', 'yhat_lower', 'yhat_upper']].to_csv("forecast_tableau.csv", index=False)

    fig = plot_plotly(model, forecast)
    fig.show()

    test = df[-3:]
    y_true = test['y'].values
    y_pred = forecast['yhat'][-3:].values

    mae = mean_absolute_error(y_true, y_pred)
    rmse = np.sqrt(mean_squared_error(y_true, y_pred))

    print(f"MAE: {mae:.2f}, RMSE: {rmse:.2f}")


if __name__ == "__main__":
    main()