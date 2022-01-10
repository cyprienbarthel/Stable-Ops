import streamlit as st
from multiapp import MultiApp
from apps import home, Ordered_Production_Data_Analysis, Construct_Times_Table,\
    Advanced_Statistics, Causes_Finder_Obs, Causes_Finder_Causal_Nex, Causes_Finder_ML_Model # import your app modules here
import pandas as pd
import shared_dataset

app = MultiApp()
#col1, col2, col3 = st.columns(3)
#with col1:
#    st.image('Data Files\IRIS_LOGO.PNG')
#with col3:
#    st.image('Data Files\OVH_Logo.PNG')
col1, col2 = st.columns(2)
with col1:
    st.markdown("## Operations Stability")
with col2:
    st.write('')
    st.write('')
    st.write (" 🚀 *made by [@Iris by Argon&Co] (https://www.irisbyargonandco.com/fr/)*")




# Add all your application here
#app.add_app("Home Page - Upload Data", home.app)
app.add_app("Upload & Transform", Construct_Times_Table.app)
app.add_app("Advanced Statistics", Advanced_Statistics.app)
app.add_app("Find delays causes with subgroups analysis", Causes_Finder_Obs.app)
app.add_app("Find delays causes with bayesian networks (Causal Nex)", Causes_Finder_Causal_Nex.app)
app.add_app("Find delays causes with ML Classification Models", Causes_Finder_ML_Model.app)

# The main app
#st.markdown('❗ **You must upload the data at first** ❗')
app.run()
#st.markdown(" ")
