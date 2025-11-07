import pandas as pd
from prophet import Prophet
from prophet.plot import plot_plotly


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

    print(forecast[['ds', 'yhat', 'yhat_lower', 'yhat_upper']])

    fig = plot_plotly(model, forecast)
    fig.show()


if __name__ == "__main__":
    main()