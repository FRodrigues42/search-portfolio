from typing import List

import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
from enum import Enum
from dateutil.relativedelta import relativedelta

pd.set_option('display.max_rows', None)
pd.set_option('display.max_columns', None)
pd.set_option('display.width', None)
pd.set_option('display.max_colwidth', None)


class Factor(Enum):
    BETA = 'Mkt-RF'
    SMB = 'SMB'
    HML = 'HML'
    RMW = 'RMW'
    CMA = 'CMA'
    MOM = 'WML'
    MKT = 'Mkt'
    RF = 'RF'
    FX = 'FX'
    USRF = 'USRF'
    USBETA = 'USMkt-USRF'
    USMKT = 'USMkt'

    def __get__(self, instance, owner):
        return self.value


def load_french_dataframe(file: str) -> pd.DataFrame:
    import re
    from io import StringIO

    with open(file) as reader:
        cleaned_csv = [line for line in reader.readlines()
                       if bool(re.search('^[,1-9]', line))]

    return pd.read_csv(
        StringIO(''.join(cleaned_csv)),
        index_col=0,
        parse_dates=[0],
        infer_datetime_format=True,
        low_memory=False
    )


def load_ecb_dataframe(file: str) -> pd.DataFrame:
    from io import StringIO

    with open(file) as reader:
        cleaned_csv = [line for line in reader.readlines()]

    return pd.read_csv(
        StringIO(''.join(cleaned_csv)),
        index_col='TIME_PERIOD',
        parse_dates=['TIME_PERIOD'],
        infer_datetime_format=True,
        low_memory=False
    )


def load_forex_dataframe(file: str) -> pd.DataFrame:
    from io import StringIO

    with open(file) as reader:
        cleaned_csv = [line for line in reader.readlines()]

    return pd.read_csv(
        StringIO(''.join(cleaned_csv)),
        index_col='DATE',
        parse_dates=['DATE'],
        infer_datetime_format=True,
        low_memory=False
    )


def load_europe_rf_data() -> pd.DataFrame:
    # https://fred.stlouisfed.org/series/IR3TIB01DEM156N
    germany_data = load_french_dataframe('data/germany/IR3TIB01DEM156N.csv')

    # https://sdw.ecb.europa.eu/quickview.do?SERIES_KEY=143.FM.M.U2.EUR.RT.MM.EURIBOR3MD_.HSTA
    euribor_data = load_ecb_dataframe('data/euribor/data.csv')
    for c in list(euribor_data.columns.array):
        if c != 'OBS_VALUE':
            euribor_data.pop(c)

    germany_data = germany_data.rename(columns={'RATE': Factor.RF})
    euribor_data = euribor_data.rename(columns={'OBS_VALUE': Factor.RF})

    return pd.concat([germany_data, euribor_data])


def load_french_developed_factor_data() -> pd.DataFrame:
    data = load_french_dataframe(
        'data/french/Developed_5_Factors_Daily.csv').merge(
        load_french_dataframe('data/french/Developed_MOM_Factor_Daily.csv'),
        left_index=True, right_index=True)

    data.attrs['group'] = 'developed'

    return data


def load_french_europe_factor_data() -> pd.DataFrame:
    data = load_french_dataframe(
        'data/french/Europe_5_Factors_Daily.csv').merge(
        load_french_dataframe('data/french/Europe_MOM_Factor_Daily.csv'),
        left_index=True, right_index=True)

    data.attrs['group'] = 'europe'

    return data


def load_french_usa_factor_data() \
        -> pd.DataFrame:
    data = load_french_dataframe(
        'data/french/F-F_Research_Data_5_Factors_2x3_daily.CSV').merge(
        load_french_dataframe('data/french/F-F_Momentum_Factor_daily.CSV'),
        left_index=True, right_index=True)

    data[Factor.MOM] = data['Mom   ']
    data.pop('Mom   ')

    data.attrs['group'] = 'usa'

    return data


def market_days_for_each_year(dates_index: pd.Index) -> List[int]:
    dates_array_aux = dates_index.copy()

    market_days_for_each_year_list = []
    for initial_index_date in dates_index:
        initial_date = pd.to_datetime(initial_index_date)
        delta = relativedelta(years=1)

        end_date = initial_date + delta
        if end_date in dates_index:
            number_days = len(dates_array_aux[(
                    (dates_array_aux >= initial_index_date) &
                    (dates_array_aux <= end_date)
            )])

            market_days_for_each_year_list.append(number_days)
            dates_array_aux = np.delete(dates_array_aux, 0)

    return market_days_for_each_year_list


def calculate_average_market_days_in_a_year(
        factor_data: pd.DataFrame,
        load: bool = True,
        save: bool = True
) -> float:
    from pathlib import Path

    if load is True and factor_data.attrs['group'] is not None:
        average_market_days_in_a_year_file = \
            Path('data/average-market-days-in-a-year/' +
                 factor_data.attrs['group'] + '.txt')
        if average_market_days_in_a_year_file.is_file():
            with open(average_market_days_in_a_year_file) as reader:
                return float(reader.readline())

    market_days_for_each_year_list = market_days_for_each_year(
        factor_data.index
    )

    average_market_days_in_a_year = np.average(market_days_for_each_year_list)

    if save is True and factor_data.attrs['group'] is not None:
        with open('data/average-market-days-in-a-year/' +
                  factor_data.attrs['group'] + '.txt', 'w+') as file:
            file.write(repr(average_market_days_in_a_year) + '\n')

    return average_market_days_in_a_year


def add_forex(factor_data: pd.DataFrame) -> pd.DataFrame:
    # https://sdw.ecb.europa.eu/quickview.do?SERIES_KEY=120.EXR.D.USD.EUR.SP00.A
    euro_usd_forex_data = load_ecb_dataframe('data/forex/data.csv') \
        [['OBS_VALUE']].copy()

    # https://www.ecb.europa.eu/press/pr/date/1998/html/pr981231_2.en.html
    mark_euro = 1.95583
    euro_mark = 1 / mark_euro

    euro_usd_forex_data['EXEUUS'] = 1 / euro_usd_forex_data['OBS_VALUE']
    euro_usd_forex_data.pop('OBS_VALUE')
    for i in range(1, len(euro_usd_forex_data)):
        if np.isnan(euro_usd_forex_data.iat[i, 0]):
            euro_usd_forex_data.iat[i, 0] = euro_usd_forex_data.iat[i - 1, 0]

    # https://fred.stlouisfed.org/series/EXGEUS
    # I supposed this data is in the beginning of each month
    euro_usd_forex_earlier_monthly_data = \
        load_forex_dataframe('data/forex/EXGEUS.csv')

    euro_usd_forex_earlier_monthly_data['EXEUUS'] = \
        euro_usd_forex_earlier_monthly_data['EXGEUS'] * euro_mark
    euro_usd_forex_earlier_monthly_data.pop('EXGEUS')

    factor_data[Factor.FX] = np.NAN

    return populate_fx(
        factor_data,
        euro_usd_forex_data,
        euro_usd_forex_earlier_monthly_data
    )


def populate_fx(
        factor_data: pd.DataFrame,
        euro_usd_forex_data: pd.DataFrame,
        euro_usd_forex_earlier_monthly_data: pd.DataFrame
) -> pd.DataFrame:
    fx_i = 0
    fx_j = 0
    previous_fx = 0
    for row in factor_data.itertuples():
        # - 1 because data has 1999-01-01
        if fx_i < len(euro_usd_forex_earlier_monthly_data.index) - 1:
            fx_i, previous_fx = populate_earlier_monthly_fx(
                euro_usd_forex_earlier_monthly_data, factor_data, fx_i,
                previous_fx, row)
        else:
            fx_j, previous_fx = populate_later_fx(euro_usd_forex_data,
                                                  factor_data,
                                                  fx_j, previous_fx, row)

    return factor_data


def populate_later_fx(
        row,
        factor_data,
        euro_usd_forex_data,
        fx_j,
        previous_fx
):
    if row.Index < euro_usd_forex_data.index[fx_j]:
        factor_data.at[row.Index, Factor.FX] = previous_fx
    elif row.Index == euro_usd_forex_data.index[fx_j]:
        factor_data.at[row.Index, Factor.FX] = euro_usd_forex_data \
            .iat[fx_j, 0]
        fx_j += 1
    else:
        while row.Index > euro_usd_forex_data.index[fx_j]:
            fx_j += 1

        if row.Index < euro_usd_forex_data.index[fx_j]:
            factor_data.at[row.Index, Factor.FX] = previous_fx
        else:
            factor_data.at[row.Index, Factor.FX] = euro_usd_forex_data \
                .iat[fx_j, 0]
            fx_j += 1
    previous_fx = factor_data.at[row.Index, Factor.FX]
    return fx_j, previous_fx


def populate_earlier_monthly_fx(
        row,
        factor_data,
        euro_usd_forex_earlier_monthly_data,
        fx_i,
        previous_fx
):
    factor_data.at[row.Index, Factor.FX] = \
        euro_usd_forex_earlier_monthly_data.iat[fx_i, 0]
    if row.Index.month != euro_usd_forex_earlier_monthly_data \
            .index[fx_i].month:
        previous_fx = factor_data.at[row.Index, Factor.FX]
        fx_i += 1
    return fx_i, previous_fx


def calculate_mkt_with_forex(factor_data: pd.DataFrame) -> pd.DataFrame:
    mkt_i = factor_data.columns.get_loc(Factor.MKT)
    fx_i = factor_data.columns.get_loc(Factor.FX)
    for i in range(1, len(factor_data)):
        euro_prev_value = factor_data.iat[i - 1, fx_i]

        euro_gains = (1 + factor_data.iat[i, mkt_i] / 100) * \
                     factor_data.iat[i, fx_i]

        factor_data.iat[i, mkt_i] = (euro_gains / euro_prev_value - 1) * 100

    return factor_data


def europeanize(
        factor_data: pd.DataFrame,
        eu_rf_data: pd.DataFrame,
        model_factor_data: pd.DataFrame = None
) -> pd.DataFrame:
    if model_factor_data is not None:
        data = factor_data[(
                (factor_data.index >= model_factor_data.index[0]) &
                (factor_data.index <= model_factor_data.index[-1])
        )].copy()
        data.attrs['group'] = factor_data.attrs['group']
        factor_data = data

    empty_factor_data = pd.DataFrame(index=factor_data.index)
    empty_factor_data[Factor.USRF] = factor_data[Factor.RF]
    empty_factor_data[Factor.RF] = factor_data[Factor.RF]

    market_days = calculate_average_market_days_in_a_year(factor_data)

    eu_rf_i = 0
    annualized_rf = eu_rf_data.iat[eu_rf_i, 0] / 100
    for row in empty_factor_data.itertuples():
        if row.Index.month > eu_rf_data.index[eu_rf_i].month or \
                row.Index.year > eu_rf_data.index[eu_rf_i].year:
            eu_rf_i += 1
            annualized_rf = eu_rf_data.iat[eu_rf_i, 0] / 100

        rf = (1 + annualized_rf) ** (1 / market_days) - 1

        empty_factor_data.at[row.Index, Factor.RF] = rf * 100

    factor_data[Factor.USRF] = factor_data[Factor.RF]
    factor_data[Factor.RF] = empty_factor_data[Factor.RF]

    factor_data[Factor.USBETA] = factor_data[Factor.BETA]
    factor_data[Factor.BETA] += factor_data[Factor.USRF]
    factor_data[Factor.MKT] = factor_data[Factor.BETA]
    factor_data[Factor.USMKT] = factor_data[Factor.MKT]

    factor_data = add_forex(factor_data)

    factor_data = calculate_mkt_with_forex(factor_data)

    # Calculate Beta
    factor_data[Factor.BETA] = factor_data[Factor.MKT]
    for row in factor_data.itertuples():
        factor_data.at[row.Index, Factor.BETA] = round(
            factor_data.at[row.Index, Factor.BETA] -
            factor_data.at[row.Index, Factor.RF],
            3
        )

    return format_factor_data(factor_data)


def format_factor_data(factor_data):
    rf_col = factor_data.pop(Factor.RF)
    mkt_col = factor_data.pop(Factor.MKT)
    fx_col = factor_data.pop(Factor.FX)
    us_beta_col = factor_data.pop(Factor.USBETA)
    us_mkt_col = factor_data.pop(Factor.USMKT)
    us_rf_col = factor_data.pop(Factor.USRF)

    factor_data.insert(len(factor_data.columns), Factor.RF, rf_col)
    factor_data.insert(len(factor_data.columns), Factor.MKT, mkt_col)
    factor_data.insert(len(factor_data.columns), Factor.FX, fx_col)
    factor_data.insert(len(factor_data.columns), Factor.USBETA, us_beta_col)
    factor_data.insert(len(factor_data.columns), Factor.USRF, us_rf_col)
    factor_data.insert(len(factor_data.columns), Factor.USMKT, us_mkt_col)

    return factor_data


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


europe_rf_data = load_europe_rf_data()
developed_factor_data = europeanize(
    load_french_developed_factor_data(), europe_rf_data
)
europe_factor_data = europeanize(
    load_french_europe_factor_data(), europe_rf_data
)
usa_factor_data = europeanize(
    load_french_usa_factor_data(),
    europe_rf_data,
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
