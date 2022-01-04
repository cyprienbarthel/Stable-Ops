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
    df_times = shared_dataset.df_times
    time_cols = shared_dataset.time_cols
    all_data = shared_dataset.all_data
    #dates_col = shared_dataset.dates_col
    #waiting_time_cols = shared_dataset.waiting_time_cols

    st.write('### Basics Statistics')

    #st.write('###### Mean of Time Variables')
    df_mean = df_times.mean().to_frame().reset_index().rename(columns={'index': 'Times', 0: 'Mean'})
    fig = px.bar(df_mean, x='Times', y='Mean', color='Mean', height=400)
    #st.plotly_chart(fig, use_container_width=True)

    st.write('###### Mean and Median of Time Variables')
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

    time_cols_without_totals = []
    prod_time_cols = []
    for i in time_cols:
        if 'Total' not in i:
            time_cols_without_totals.append(i)
            if 'Log' not in i: #### A changer pour généraliser
                prod_time_cols.append(i)

    fig = px.imshow(df_times[time_cols_without_totals].corr())
    st.plotly_chart(fig, use_container_width=True)
    _cov_perc = round(abs(df_times[prod_time_cols].var().sum() - \
                          df_times['Total Time Prod'].var()) / df_times['Total Time Prod'].var() * 100, 1)

    st.markdown(f"Covariance part of Prod Variance is {_cov_perc}%")


    st.write(' ')
    st.write('')
    st.write('### Statistical Distribution of Time Variables')

    TS_to_Analyse = st.multiselect("Select Time to Analyse", time_cols + ['Total Time'])

    dico_dist = {'Normale': norm, 'Log Normale': lognorm, 'Chi2': chi2, 'Truncated Normale': truncnorm,
                 'Log Uniform': loguniform}

    if len(TS_to_Analyse) > 0:
        Stab_analyser2 = StabAnalyser(all_data)
        TS_to_Analyse = TS_to_Analyse[0]
        # fig_hist = Stab_analyser2.histogram_plot(TS_to_Analyse)
        # st.plotly_chart(fig_hist, use_container_width=True)
        fig_dens = Stab_analyser2.density_plots(TS_to_Analyse)
        st.pyplot(fig_dens[0], use_container_width=True)  # , stats.norm)#, floc=0)

        law = st.selectbox("Test to Fit a Law", ('Normale', 'Log Normale', 'Chi2', 'Truncated Normale', 'Log Uniform'))
        law = dico_dist[law]
        if st.button(' ✅ Fit Distribution'):

            fig1, fig2, squared_estimate_errors, aic, dist_params = Stab_analyser2.fit_stat_law(TS_to_Analyse, law)
            col1, col2 = st.columns(2)
            with col1:
                st.pyplot(fig1, use_container_width=True)
            with col2:
                st.pyplot(fig2, use_container_width=True)

            st.markdown(f'AIC {aic}')
            st.markdown(f'RMSE {squared_estimate_errors}')
            st.dataframe(pd.DataFrame.from_dict(dist_params).rename(columns={0:'Params'}))




