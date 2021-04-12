import pandas as pd
import re
import numpy as np
from io import StringIO

from utils import Factor


def load_french_dataframe(file: str) -> pd.DataFrame:
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


def load_forex_data():
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
    # I suppose this data is in the beginning of each month
    euro_usd_forex_earlier_monthly_data = \
        load_forex_dataframe('data/forex/EXGEUS.csv')

    euro_usd_forex_earlier_monthly_data['EXEUUS'] = \
        euro_usd_forex_earlier_monthly_data['EXGEUS'] * euro_mark

    euro_usd_forex_earlier_monthly_data.pop('EXGEUS')

    return euro_usd_forex_data, euro_usd_forex_earlier_monthly_data
