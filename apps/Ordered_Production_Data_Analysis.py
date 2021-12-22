import streamlit as st
import pandas as pd
from datetime import datetime, timedelta,date
import plotly.express as px
import numpy as np

import sys
sys.path.append('.')
from Stable_Ops_Functions import StabAnalyser, CausalAnalyser
import matplotlib.pyplot as plt
import seaborn as sns
from scipy.stats import chi2, norm, poisson, lognorm, truncnorm, loguniform

def app():
    #st.title('Stability Analysis of Serveurs Production')
    st.write('')
    st.write('')
    st.write('### Uploading Dataset')

    df_path = st.file_uploader("Upload a Dataset", type=["xlsx"])
    #@st.cache(suppress_st_warning=True)
    def upload_data():
        df = pd.read_excel(df_path, skiprows=1)
        #df = pd.read_excel('Data Files\DhoDvUCA.xlsx', skiprows=1)
        return df

    if df_path is not None:
        df = upload_data()
        st.dataframe(df)

        start_date = st.sidebar.date_input('start date', date(2021, 7, 1))
        start_date = pd.to_datetime(start_date)


        def construct_data_table(df):
            df['Start'] = pd.to_datetime(df['Start'])
            df = df[df['Start'] > start_date]
            df['Supposed Task Order'] = df['Supposed Task Order'].apply(str)
            df_pivoted = df.pivot(index='OF', columns='Supposed Task Order', values=['Start', 'End'])
            df_pivoted.columns = df_pivoted.columns.map('|'.join).str.strip('|')
            #st.dataframe(df_pivoted)
            return df_pivoted

        working_hours = st.sidebar.number_input('Working Hours in a Day', min_value=0, max_value=24, value=9, step=1)

        not_working_hours = 24 - working_hours

        def get_actual_exec_time(date_fin, date_deb,not_working_hours=not_working_hours):
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

        df_pivoted = construct_data_table(df)

        df_pivoted = get_times(df_pivoted)

        tole = st.sidebar.number_input('Tolerance for Orders in minutes', min_value=0, max_value=1440, value=15, step=1)

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
        st.dataframe(df_pivoted['OF is ordered'].value_counts())

        filter = st.sidebar.selectbox(
            'Filter only properly ordered OF?',
            ('Yes','No'), 0)

        if filter == 'Yes':
            df_pivoted = df_pivoted[df_pivoted['OF is ordered']=='Yes']

        time_cols = []
        dates_col = []
        for i in range(1, tasks_number + 1):
            time_cols.append(f'Time Task|{i}')
            dates_col.append(f'End|{i}')
            dates_col.append(f'Start|{i}')
            if i != tasks_number:
                time_cols.append(f'Waiting Time {i}|{i + 1}')

        df_times = df_pivoted[time_cols]

        df_pivoted['Start|OF'] = df_pivoted[dates_col].min(axis=1)
        df_pivoted['End|OF'] = df_pivoted[dates_col].max(axis=1)
        df_pivoted['Total Time'] = df_pivoted.apply(lambda df_pivoted: \
                                                        get_actual_exec_time(df_pivoted[f'End|OF'],
                                                                             df_pivoted[f'Start|OF']),
                                                    axis=1)
        df_times['Total Time'] = df_pivoted['Total Time']



        st.write('')
        st.write('')

        st.write('### Basics Statistics')


        st.write('###### Mean of Time Variables')
        df_mean = df_times.mean().to_frame().reset_index().rename(columns={'index': 'Times', 0: 'Mean'})
        fig = px.bar(df_mean, x='Times', y='Mean', color='Mean', height=400)
        #st.plotly_chart(fig, use_container_width=True)

        st.write('###### Median of Time Variables')
        df_median = df_times.median().to_frame().reset_index().rename(columns={'index': 'Times', 0: 'Median'})
        fig2 = px.bar(df_median, x='Times', y='Median', color='Median', height=400)
        #st.plotly_chart(fig, use_container_width=True)

        col1, col2 = st.columns(2)
        with col1:
            st.plotly_chart(fig, use_container_width=True)
        with col2:
            st.plotly_chart(fig2, use_container_width=True)

        st.write('###### Standard Deviation of Time Variables')
        df_std = df_times.std().to_frame().reset_index().rename(columns={'index':'Times', 0:'Standard Deviation'})
        fig = px.bar(df_std, x='Times', y='Standard Deviation', color='Standard Deviation', height=400)
        st.plotly_chart(fig, use_container_width=True)

        st.write('###### Autocorrelation between Time Variables')
        fig = px.imshow(df_times[time_cols].corr())
        st.plotly_chart(fig, use_container_width=True)
        _cov_perc = round(abs(df_times[time_cols].var().sum() - \
                              df_times['Total Time'].var()) / df_times['Total Time'].var() * 100, 1)

        st.markdown(f"Covariance part of the variance is {_cov_perc}%")

        st.write(' ')
        st.write('')
        st.write('### Statistical Distribution of Time Variables')

        TS_to_Analyse = st.multiselect("Select Time to Analyse", time_cols + ['Total Time'])

        dico_dist = {'Normale': norm, 'Log Normale': lognorm, 'Chi2': chi2, 'Truncated Normale': truncnorm,
                     'Log Uniform': loguniform}

        if len(TS_to_Analyse) > 0:
            Stab_analyser2 = StabAnalyser(df_pivoted)
            TS_to_Analyse = TS_to_Analyse[0]
            # fig_hist = Stab_analyser2.histogram_plot(TS_to_Analyse)
            # st.plotly_chart(fig_hist, use_container_width=True)
            fig_dens = Stab_analyser2.density_plots(TS_to_Analyse)
            st.pyplot(fig_dens[0], use_container_width=True)  # , stats.norm)#, floc=0)

            law = st.selectbox("Test to Fit a Law", ('Normale', 'Log Normale', 'Chi2', 'Truncated Normale', 'Log Uniform'))
            law = dico_dist[law]
            if st.button('Fit Distribution'):

                fig1, fig2, squared_estimate_errors, aic, dist_params = Stab_analyser2.fit_stat_law(TS_to_Analyse, law)
                col1, col2 = st.columns(2)
                with col1:
                    st.pyplot(fig1, use_container_width=True)
                with col2:
                    st.pyplot(fig2, use_container_width=True)

                st.markdown(f'AIC {aic}')
                st.markdown(f'RMSE {squared_estimate_errors}')
                st.dataframe(pd.DataFrame.from_dict(dist_params).rename(columns={0:'Params'}))

        st.write(' ')
        st.write('')
        st.write('### Causes Finder 1 : Analyse of Tasks Times Distribution by groups')

        #df_to_analyse = st.sidebar.selectbox(
        #    'Analyse Tasks Times Detail or Total OF Times',
        #    ('Tasks Times Detail', 'Total OF Times'), 0)


        ### Add Times Attributes
        df['Week Day'] = df['Start'].apply(lambda x: x.weekday()).replace(
            {0: '1.Lundi', 1: '2.Mardi', 2: '3.Mercredi', 3: '4.Jeudi', 4: '5.Vendredi', 5: '6.Samedi'})
        df['Hour'] = df['Start'].apply(lambda x: str(x.hour))

        df['Task Time'] = df.apply(lambda df: get_actual_exec_time(df[f'End'],df[f'Start']),axis=1)


        ### Initiate StabAnalyser

        Stab_analyser = StabAnalyser(df)

        #time_variable = st.select("Time Variable to analyse", time_cols)
        possibles_groups = set(df.columns.to_list()) - set(['Task Time', 'OF', 'End', 'Start','Supposed Task Order'])
        group_variable = st.multiselect("Group to analyse", possibles_groups)
        int_val = st.slider('Max value of histograms', min_value=0, max_value=100, value=30, step=2)
        bin_size = st.slider('Bin size of density Histogram', min_value=0.1, max_value=5.0, value=0.2, step=0.1)
        if len(group_variable) > 0:
            group_variable = group_variable[0]
            st.write(group_variable)
            fig1, fig2 = Stab_analyser.group_analyser(group_variable, 'Task Time', int_val,bin_size)

        else:
            fig1, fig2 = Stab_analyser.group_analyser(np.nan, 'Task Time', int_val, bin_size)

        st.plotly_chart(fig2, use_container_width=True)

        st.plotly_chart(fig1, use_container_width=True)

        st.write(' ')
        st.write('')
        st.write('### Causes Finder 2 : Causal Model')

        causal_analyser = CausalAnalyser(df)

        causes_VA = st.multiselect("Possible Causes", possibles_groups)
        tabu_child = st.multiselect("Nodes that are not a consequence", possibles_groups)

        if len(causes_VA) > 0 :
            fig, ie, probas = causal_analyser.causes_finder(
                features=causes_VA,
                target='Task Time', edge_value_tresh=0,
                tabu_child_nodes=tabu_child)
            st.pyplot(fig, use_container_width=True)
            st.dataframe(probas)









