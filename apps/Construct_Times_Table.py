import streamlit as st
import pandas as pd
from datetime import datetime, timedelta,date
import plotly.express as px
import numpy as np
import shared_dataset

import sys
sys.path.append('.')
from Stable_Ops_Functions import StabAnalyser, CausalAnalyser, CausalAnalyser_v2
import matplotlib.pyplot as plt
import seaborn as sns
from scipy.stats import chi2, norm, poisson, lognorm, truncnorm, loguniform

def app():
    #st.title('Stability Analysis of Serveurs Production')
    st.write('')
    st.write('')
    #df = upload_data()
    df = shared_dataset.input_dataset
    df_carac = shared_dataset.carac_dataset #pd.read_excel('Data Files\Data_OVH_2.xlsx', sheet_name=1)  #### Path changer
    st.markdown('Input Table ⬇️')
    st.dataframe(df)
    start_date = st.sidebar.date_input('start date', date(2021, 7, 1))
    start_date = pd.to_datetime(start_date)
    filter_query = st.sidebar.text_input("Filter Query", '')
    if filter_query != '':
        df = df.query(filter_query)

    def construct_ordered_data_table(df):
        df = df[df['Supposed Task Order'] != -1]
        df['Start'] = pd.to_datetime(df['Start'])
        df = df[df['Start'] > start_date]
        df['Supposed Task Order'] = df['Supposed Task Order'].apply(str)
        df_pivoted = df.pivot(index='OF', columns='Supposed Task Order', values=['Start', 'End'])
        df_pivoted.columns = df_pivoted.columns.map('|'.join).str.strip('|')
        #st.dataframe(df_pivoted)
        return df_pivoted

    working_hours = st.sidebar.number_input('Working Hours in a Day', min_value=0, max_value=24, value=9, step=1)

    not_working_hours = 24 - working_hours

    def get_actual_exec_time(date_fin, date_deb, not_working_hours = not_working_hours):
        if date_fin != date_fin or date_deb != date_deb:
            return np.nan
        else:
            brut_delta = (date_fin - date_deb).days * 24 + (date_fin - date_deb).seconds / 3600  ## in hours
            delta_days = (date_fin.date() - \
                          date_deb.date()).days
            week_end = (date_fin.weekday() < date_fin.weekday()) * 1

        return brut_delta - delta_days * not_working_hours - week_end * 24 * 2  ## Number of not worked hours


    tasks_number = df['Supposed Task Order'].max()

    def get_times(df_pivoted):

        for i in range(1, tasks_number+1):
            df_pivoted[f'Time Task|{i}'] =  df_pivoted.apply(lambda df_pivoted: \
                                        get_actual_exec_time(df_pivoted[f'End|{i}'],
                                                             df_pivoted[f'Start|{i}']),
                                                             axis=1)
            if i != tasks_number:

                df_pivoted[f'Waiting Time {i}|{i+1}'] = df_pivoted.apply(lambda df_pivoted: \
                                                                     get_actual_exec_time(df_pivoted[f'Start|{i+1}'],
                                                                                          df_pivoted[f'End|{i}']),
                                                                    axis=1)
        return df_pivoted

    df_pivoted = construct_ordered_data_table(df)

    df_pivoted = get_times(df_pivoted)



    tole = st.sidebar.number_input('Tolerance for Orders in minutes', min_value=0,
                                   max_value=1440, value=15, step=1)

    tole = timedelta(minutes=1)

    def check_order(df_pivoted, tole = tole):
        a = 1
        for i in range(1, tasks_number-1):
            a = a * ((df_pivoted[f'Start|{i+1}'] - df_pivoted[f'End|{i}']) >= -tole)*1
        if a == 1:
            return 'Yes'
        else:
            return 'No'


    df_pivoted['OF is ordered'] = df_pivoted.apply(check_order, axis=1)
    #st.dataframe(df_pivoted['OF is ordered'].value_counts())

    filter = st.sidebar.selectbox(
        'Filter only properly ordered OF?',
        ('Yes','No'), 0)

    if filter == 'Yes':
        df_pivoted = df_pivoted[df_pivoted['OF is ordered']=='Yes']

    time_cols = []
    dates_col = []
    waiting_time_cols = []
    for i in range(1, tasks_number + 1):
        time_cols.append(f'Time Task|{i}')
        dates_col.append(f'End|{i}')
        dates_col.append(f'Start|{i}')
        if i != tasks_number:
            time_cols.append(f'Waiting Time {i}|{i + 1}')
            waiting_time_cols.append(f'Waiting Time {i}|{i + 1}')

    df_pivoted['Total Waiting Time'] = df_pivoted[waiting_time_cols].sum(axis=1)
    time_cols.append('Total Waiting Time')

    df_pivoted['Start|Prod OF'] = df_pivoted[dates_col].min(axis=1)
    df_pivoted['End|Prod OF'] = df_pivoted[dates_col].max(axis=1)
    df_pivoted['Total Time Prod'] = df_pivoted.apply(lambda df_pivoted: \
                                                         get_actual_exec_time(df_pivoted['End|Prod OF'],
                                                                              df_pivoted['Start|Prod OF']),
                                                     axis=1)

    time_cols.append('Total Time Prod')
    dates_col.append('Start|Prod OF')
    dates_col.append('End|Prod OF')



    def construct_not_ordered_data_table(df):
        df__1 = df[df['Supposed Task Order'] == -1]
        df__1['Start'] = pd.to_datetime(df__1['Start'])
        df__1 = df__1[df__1['Start'] > start_date]
        df_pivoted_1 = df__1.pivot(index='OF', columns='Task type', values=['Start', 'End'])
        df_pivoted_1.columns = df_pivoted_1.columns.map('|'.join).str.strip('|')
        # st.dataframe(df_pivoted)
        return df_pivoted_1

    log_data = construct_not_ordered_data_table(df)
    not_ordered_tasks = df[df['Supposed Task Order'] == -1]['Task type'].unique()

    for i in not_ordered_tasks:
        log_data[f'Time Task|{i}'] = log_data.apply(lambda log_data: \
                         get_actual_exec_time(log_data[f'End|{i}'],
                                              log_data[f'Start|{i}']),
                            axis=1)

        time_cols.append(f'Time Task|{i}')
        dates_col.append(f'End|{i}')
        dates_col.append(f'Start|{i}')

    all_data = df_pivoted.merge(log_data, on ='OF', how = 'inner')

    all_data['Start|OF'] = all_data[dates_col].min(axis=1)
    all_data['End|OF'] = all_data[dates_col].max(axis=1)
    all_data['Total Time OF'] = all_data.apply(lambda all_data: \
                                                         get_actual_exec_time(all_data['End|OF'],
                                                                              all_data['Start|OF']),
                                                     axis=1)

    time_cols.append('Total Time OF')
    dates_col.append('Start|OF')
    dates_col.append('End|OF')
    df_times = all_data[time_cols]

    all_data.to_excel('df_pivoted_table.xlsx')

    st.write('')
    st.write('')


    c_to_str = list(set(df_carac.columns.to_list()) - set(['OF']))
    df_carac[c_to_str] = df_carac[c_to_str].applymap(str)
    all_data_merged = all_data.merge(df_carac, on='OF', how='left')

    ### Add Times Attributes
    all_data_merged['Start Week Day'] = all_data_merged['Start|OF'].apply(lambda x: x.weekday()).replace(
        {0: '1.Lundi', 1: '2.Mardi', 2: '3.Mercredi', 3: '4.Jeudi', 4: '5.Vendredi', 5: '6.Samedi'})
    all_data_merged['Start Hour'] = all_data_merged['Start|OF'].apply(lambda x: str(x.hour))

    shared_dataset.all_data = all_data
    shared_dataset.all_data_merged = all_data_merged
    shared_dataset.df_times = df_times
    shared_dataset.time_cols = time_cols
    shared_dataset.dates_col = dates_col
    shared_dataset.waiting_time_cols = waiting_time_cols

    st.markdown('Output Table ⬇️')
    st.dataframe(all_data_merged)
    st.markdown('✅ Output Table is saved')