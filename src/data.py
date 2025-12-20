"""
Modul untuk memuat dan memproses data populasi.
"""

import io
import pandas as pd
import streamlit as st
from typing import Tuple


@st.cache_data
def load_data() -> pd.DataFrame:
    """Muat dataset populasi Moose dan Serigala dari string.

    Returns:
        pd.DataFrame: DataFrame dengan kolom 'Year', 'Wolves', 'Moose'.
    """
    data_string = """
    Year,Wolves,Moose
    1980,50,664
    1981,30,650
    1982,14,700
    1983,23,900
    1984,24,811
    1985,22,1062
    1986,20,1025
    1987,16,1380
    1988,12,1653
    1989,11,1397
    1990,15,1216
    1991,12,1313
    1992,12,1600
    1993,13,1880
    1994,15,1800
    1995,16,2400
    1996,22,1200
    1997,24,500
    1998,14,700
    1999,25,750
    2000,29,850
    2001,19,900
    2002,17,1000
    2003,19,900
    2004,29,750
    2005,30,540
    2006,30,385
    2007,21,450
    2008,23,650
    2009,24,530
    2010,19,510
    2011,16,515
    2012,9,750
    2013,8,975
    2014,9,1050
    2015,3,1250
    2016,2,1300
    2017,2,1600
    2018,2,1500
    2019,14,2060
    """
    df = pd.read_csv(io.StringIO(data_string.strip()))
    df = df.set_index('Year')
    return df

def filter_data_by_year(df: pd.DataFrame, year_range: Tuple[int, int]) -> pd.DataFrame:
    """Filter DataFrame berdasarkan rentang tahun yang dipilih.

    Args:
        df (pd.DataFrame): DataFrame asli.
        year_range (Tuple[int, int]): Tuple berisi tahun mulai dan tahun akhir.

    Returns:
        pd.DataFrame: DataFrame yang sudah difilter.
    """
    start_year, end_year = year_range
    return df.loc[start_year:end_year]
