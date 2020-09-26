import pandas as pd
from fbprophet import Prophet
import matplotlib.pyplot as plt

df = pd.read_csv("passengers.csv")
df.head()

m = Prophet()
m.fit(df)

future = m.make_future_dataframe(periods= 24, freq = 'MS')
future.tail()

forecast = m.predict(future)
forecast[['ds', 'yhat', 'yhat_lower', 'yhat_upper']].tail()

fig1 = m.plot(forecast)
fig2 = m.plot_components(forecast)

from fbprophet.plot import plot_plotly, plot_components_plotly
plot_plotly(m, forecast)

plot_components_plotly(m, forecast)

#m.plot(forecast)
plt.show()
