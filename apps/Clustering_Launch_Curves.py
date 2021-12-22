import streamlit as st
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import re

def app():
    st.title('Clustering of Launches')

    st.write("This is a page to visualize the different types of launches")

    df_coeffs = pd.read_excel('Data Files/cluster_regime.xlsx')

    fig0, ax0 = plt.subplots(figsize=(12, 8))
    df_coeffs['Curve Type'].value_counts(normalize=True).plot(kind='bar')
    plt.show()
    st.pyplot(fig0)
    st.image('Data Files\launch_types.JPG')

    launch_types = list(df_coeffs['Curve Type'].drop_duplicates().values)
    launch_type = st.selectbox('Type of Launch', launch_types)
    dic = {'log_type': 'logarithmic', 'exp_type': 'exponential', 'slow_launch': 'slow start', 'pallier': 'landing'}
    type__ = dic[launch_type]
    st.markdown(f"Below the commercial lines with a `{type__}` launch type")
    nb_graphs = st.sidebar.slider('Nb of graphs to plot', 0, 100, 10, 1)

    def Convert(string):
        li = re.split(' |  |   |\*|\n', string[1:-1])
        l = []
        for i in li:
            if i != '':
                l.append(float(i))
        # li = list(string[1:-1].replace('\n','').split(" "))#.remove([''])
        return l




    n = df_coeffs[df_coeffs['Curve Type'] == launch_type].shape[0]
    n_tot = df_coeffs.shape[0]
    perc = round(n * 100 / n_tot, 2)
    st.markdown(f"There are `{n}` ({perc} %) commercial lines with this type of launch")
    for i in df_coeffs[df_coeffs['Curve Type'] == launch_type][:nb_graphs].index:
        print(i)
        name = df_coeffs.loc[i]['F&A Commercial Line']
        st.markdown(' ')
        st.markdown('------------------------------------------')
        st.markdown(' ')
        st.markdown(f'Commercial Line : **{name}**')
        st.markdown(' ')
        a = pd.DataFrame(df_coeffs.loc[i][['Introduction', 't1', 'Growth', 't2', 'Maturity', 'R2']].apply(
            lambda x: round(float(x), 2))) \
            .rename(columns={i: 'Parameters'})
        a['Unit'] = ['sales qty by week', 'weeks', 'sales qty by week', 'weeks', 'sales qty by week', 'None']


        fig, ax = plt.subplots(figsize=(12, 8))
        x = Convert(df_coeffs.loc[i]['x'])
        y = Convert(df_coeffs.loc[i]['y'])
        y_fitted = Convert(df_coeffs.loc[i]['y_fitted'])

        t1 = int(round(df_coeffs.loc[i]['t1'], 0))
        t2 = int(round(df_coeffs.loc[i]['t2'], 0))

        plt.scatter(x, y, label="Original Data")
        plt.plot(x, y_fitted, 'r-', label="Fitted Curve Droite")

        if t1 not in x:
            t1 = x[int((np.abs(np.array(x) - t1)).argmin())]
        if t2 not in x:
            t2 = x[int((np.abs(np.array(x) - t2)).argmin())]

        t1_index = x.index(t1)
        t2_index = x.index(t2)

        plt.scatter([0, t1, t2], [0, y_fitted[t1_index], y_fitted[t2_index]], s=[150, 150, 150], c=['r', 'r', 'r'])

        plt.legend()
        st.pyplot(fig)

        st.markdown(f'Parameters')
        st.write(a)