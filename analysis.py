import pandas as pd
import matplotlib.pyplot as plt
import numpy as np

from utils import Factor
import dataio as dio
import europeanize as euz

pd.set_option('display.max_rows', None)
pd.set_option('display.max_columns', None)
pd.set_option('display.width', None)
pd.set_option('display.max_colwidth', None)


def make_portfolio(factor_data: pd.DataFrame,
                   initial_value: float = 100,
                   factor: Factor = Factor.MKT) -> pd.DataFrame:
    portfolio_data = pd.DataFrame(index=factor_data.index)
    portfolio_data['V'] = np.NAN

    previous_value = initial_value
    for row in factor_data.itertuples():
        portfolio_data.at[row.Index, 'V'] = previous_value + \
                                            previous_value * \
                                            (factor_data.at[
                                                 row.Index, factor] / 100)
        previous_value = portfolio_data.at[row.Index, 'V']

    return portfolio_data


europe_rf_data = dio.load_europe_rf_data()
euro_usd_forex_data, euro_usd_forex_earlier_monthly_data = \
    dio.load_forex_data()

developed_factor_data = euz.europeanize(
    dio.load_french_developed_factor_data(),
    europe_rf_data,
    euro_usd_forex_data,
    euro_usd_forex_earlier_monthly_data
)

europe_factor_data = euz.europeanize(
    dio.load_french_europe_factor_data(),
    europe_rf_data,
    euro_usd_forex_data,
    euro_usd_forex_earlier_monthly_data
)

usa_factor_data = euz.europeanize(
    dio.load_french_usa_factor_data(),
    europe_rf_data,
    euro_usd_forex_data,
    euro_usd_forex_earlier_monthly_data,
    europe_factor_data
)


print("Developed")
print(developed_factor_data.head())
print("...")
print(developed_factor_data.tail())
print()
print("Europe")
print(europe_factor_data.head())
print("...")
print(europe_factor_data.tail())
print()
print("USA")
print(usa_factor_data.head())
print("...")
print(usa_factor_data.tail())

usd_portfolio = make_portfolio(usa_factor_data, factor=Factor.USMKT)
print("USD")
print(usd_portfolio.tail())

usd_portfolio.plot(y=['V'], kind='line', cmap='Dark2')
# # plt.plot(developed_factor_data.index, developed_factor_data[Factor.BETA])
plt.show()

euro_portfolio = make_portfolio(usa_factor_data, 75.9626)
print("EURO")
print(euro_portfolio.tail())

euro_portfolio.plot(y=['V'], kind='line', cmap='Dark2')
plt.show()
