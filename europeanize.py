from typing import List

import pandas as pd
import numpy as np
from dateutil.relativedelta import relativedelta

from utils import Factor


def europeanize(
        factor_data: pd.DataFrame,
        eu_rf_data: pd.DataFrame,
        euro_usd_forex_data: pd.DataFrame,
        euro_usd_forex_earlier_monthly_data: pd.DataFrame,
        model_factor_data: pd.DataFrame = None
) -> pd.DataFrame:
    """ Transforms the French's factors data into european data

    Changes the percentages to address the exchange for euros and
    subtracts the euribor to the Mkt for the Mkt-RF.

    :param factor_data: main factor data from French's website
    :type factor_data: pandas.DataFrame
    :param eu_rf_data: monthly data for the risk free return for europe
        (given by euribor and before that germany yield of 3 months)
    :type eu_rf_data: pandas.DataFrame
    :param euro_usd_forex_data: daily forex Euros to US dollars
    :type euro_usd_forex_data: pandas.DataFrame
    :param euro_usd_forex_earlier_monthly_data: before euro use the
        german mark exchange to US dollar and because there was
        a fixed exchange from marks to euros then we can extrapolate
        the euro to usd forex
    :type euro_usd_forex_earlier_monthly_data: pandas.DataFrame
    :param model_factor_data: because French's data has different
        start and end dates for some types then this data is going to
        impose its start and end dates.
    :type model_factor_data: pandas.DataFrame
    :return: factor data modified to euro and europe's risk free return
    :rtype: pandas.DataFrame
    """
    if model_factor_data is not None:
        data = factor_data[(
                (factor_data.index >= model_factor_data.index[0]) &
                (factor_data.index <= model_factor_data.index[-1])
        )].copy()
        data.attrs['group'] = factor_data.attrs['group']
        factor_data = data

    factor_data = calculate_europe_rf(factor_data, eu_rf_data)

    factor_data[Factor.USBETA] = factor_data[Factor.BETA]

    factor_data[Factor.USMKT] = factor_data[Factor.USBETA] + \
                                factor_data[Factor.USRF]

    factor_data = add_forex(
        factor_data,
        euro_usd_forex_data,
        euro_usd_forex_earlier_monthly_data
    )

    factor_data = calculate_mkt_with_forex(factor_data)

    # Calculate Beta
    for row in factor_data.itertuples():
        factor_data.at[row.Index, Factor.BETA] = round(
            factor_data.at[row.Index, Factor.MKT] -
            factor_data.at[row.Index, Factor.RF],
            3
        )

    return change_col_order(factor_data)


def calculate_europe_rf(factor_data, eu_rf_data):
    factor_data[Factor.USRF] = factor_data[Factor.RF]

    market_days = calculate_average_market_days_in_a_year(factor_data)

    eu_rf_i = 0
    annualized_rf = eu_rf_data.iat[eu_rf_i, 0] / 100
    for row in factor_data.itertuples():
        if row.Index.month > eu_rf_data.index[eu_rf_i].month or \
                row.Index.year > eu_rf_data.index[eu_rf_i].year:
            eu_rf_i += 1
            annualized_rf = eu_rf_data.iat[eu_rf_i, 0] / 100

        rf = (1 + annualized_rf) ** (1 / market_days) - 1

        factor_data.at[row.Index, Factor.RF] = rf * 100

    return factor_data


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


def add_forex(
        factor_data: pd.DataFrame,
        euro_usd_forex_data: pd.DataFrame,
        euro_usd_forex_earlier_monthly_data: pd.DataFrame
) -> pd.DataFrame:
    factor_data[Factor.FX] = np.NAN

    fx_i = 0
    fx_j = 0
    previous_fx = 0
    for row in factor_data.itertuples():
        # - 1 because data has 1999-01-01
        if fx_i < len(euro_usd_forex_earlier_monthly_data.index) - 1:
            fx_i, previous_fx = populate_earlier_monthly_fx(
                row,
                factor_data,
                euro_usd_forex_earlier_monthly_data,
                fx_i,
                previous_fx
            )
        else:
            fx_j, previous_fx = populate_later_fx(
                row,
                factor_data,
                euro_usd_forex_data,
                fx_j,
                previous_fx
            )

    return factor_data


def populate_earlier_monthly_fx(
        row,
        factor_data: pd.DataFrame,
        euro_usd_forex_earlier_monthly_data: pd.DataFrame,
        fx_i: int,
        previous_fx: float
):
    factor_data.at[row.Index, Factor.FX] = \
        euro_usd_forex_earlier_monthly_data.iat[fx_i, 0]
    if row.Index.month != euro_usd_forex_earlier_monthly_data \
            .index[fx_i].month:
        previous_fx = factor_data.at[row.Index, Factor.FX]
        fx_i += 1
    return fx_i, previous_fx


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


def calculate_mkt_with_forex(factor_data: pd.DataFrame) -> pd.DataFrame:
    factor_data[Factor.MKT] = np.NAN
    mkt_i = factor_data.columns.get_loc(Factor.MKT)
    usmkt_i = factor_data.columns.get_loc(Factor.USMKT)
    fx_i = factor_data.columns.get_loc(Factor.FX)

    factor_data.iat[0, mkt_i] = factor_data.iat[0, usmkt_i]
    for i in range(1, len(factor_data)):
        euro_prev_value = factor_data.iat[i - 1, fx_i]

        euro_gains = (1 + factor_data.iat[i, usmkt_i] / 100) * \
                     factor_data.iat[i, fx_i]

        factor_data.iat[i, mkt_i] = (euro_gains / euro_prev_value - 1) * 100

    return factor_data


def change_col_order(factor_data):
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
