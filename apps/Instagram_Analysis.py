import streamlit as st
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import re
import pandas as pd

def app():
    st.title('Quick Instagram Likes Analysis')

    st.write("Plotting of Instagram Likes")

    df = pd.read_csv('C:/Users/CYB/OneDrive - Argon Consulting/Chloé - Early Signals - Code/data_insta.csv')

    l = list(df['Commercial Line'].drop_duplicates().values)

    hashtag = st.selectbox(
        'Select a hashtag', l)

    df = df[df['Commercial Line'] == hashtag]

    fig, ax = plt.subplots(figsize=(12, 8))
    plt.plot(df['nb_likes'].values)
    st.pyplot(fig)