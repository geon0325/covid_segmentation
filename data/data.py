from datetime import timedelta, datetime, date

import numpy as np
import os
import pandas as pd
pd.options.mode.chained_assignment = None

data_dir = os.path.join('./')  # todo: handle provinces

# Other countries

def _load(path, country, start_date):
    if country == 'Korea':
        country = 'Korea, South'
    elif country == 'Uae':
        country = 'United Arab Emirates'
    elif country == 'Uk':
        country = 'United Kingdom' 
    elif country == 'Southafrica':
        country = 'South Africa'
    elif country == 'Saudiarabia':
        country = 'Saudi Arabia'
    elif country == 'Dominicanrepublic':
        country = 'Dominican Republic'
    elif country == 'Costarica':
        country = 'Costa Rica'
    elif country == 'Puertorico':
        country = 'Puerto Rico'
    elif country == 'Elsalvador':
        country = 'El Salvador'
    df = pd.read_csv(path) 
    country_df = df[df['Country/Region'].str.contains(country)].sum()
    date_country_df = country_df.loc[start_date:]
    return date_country_df

def load_confirmed(country, start_date):
    confirmed_path = os.path.join(data_dir, 'time_series_covid_19_confirmed.csv')
    return _load(confirmed_path, country, start_date)

def load_recovered(country, start_date):
    recovered_path = os.path.join(data_dir, 'time_series_covid_19_recovered.csv')
    return _load(recovered_path, country, start_date)

def load_deaths(country, start_date):
    deaths_path = os.path.join(data_dir, 'time_series_covid_19_deaths.csv')
    return _load(deaths_path, country, start_date)

def load_tests(country, start_date):
    tests_path = os.path.join(data_dir, 'full-list-total-tests-for-covid-19.csv')
    df = pd.read_csv(tests_path)
    country_df = df[df['Entity'].str.contains(country)]
    idx = country_df[country_df['Date'].str.contains('Mar')].index[0] - 1
    date_country_df = country_df.loc[idx:]['Total tests']
    return date_country_df.subtract(date_country_df.shift())[1:]

def load_ir_data(country, start_date):
    recovered = load_recovered(country, start_date)
    deaths = load_deaths(country, start_date)
    confirmed = load_confirmed(country, start_date)
    confirmed = confirmed.subtract(recovered).subtract(deaths)
    
    data_df = pd.concat([confirmed, recovered, deaths], axis=1, keys=['C', 'R', 'D'], sort=False)
    return data_df



# United States

def _load_us(path, start_date):
    df = pd.read_csv(path)
    date_df = df.loc[start_date:].sum().loc[start_date:]
    return date_df

def load_confirmed_us(start_date):
    confirmed_path = os.path.join(data_dir, 'time_series_covid_19_confirmed_US.csv')
    return _load_us(confirmed_path, start_date)

def load_deaths_us(start_date):
    deaths_path = os.path.join(data_dir, 'time_series_covid_19_deaths_US.csv')
    return _load_us(deaths_path, start_date)

def load_ir_data_us(start_date):
    deaths = load_deaths_us(start_date)
    confirmed = load_confirmed_us(start_date)
    confirmed = confirmed.subtract(deaths)
    
    data_df = pd.concat([confirmed, deaths], axis=1, keys=['C', 'D'], sort=False)
    return data_df
