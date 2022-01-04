import streamlit as st
from multiapp import MultiApp
from apps import home, Ordered_Production_Data_Analysis, Construct_Times_Table,\
    Advanced_Statistics, Causes_Finder_Obs, Causes_Finder_Causal_Nex, Causes_Finder_ML_Model # import your app modules here
import pandas as pd
import shared_dataset

app = MultiApp()
col1, col2, col3 = st.columns(3)
with col1:
    st.image('Data Files\IRIS_LOGO.PNG')
with col3:
    st.image('Data Files\OVH_Logo.PNG')

st.markdown("""
## Operations Stability
###### 🚀 Data Visualization App made by [@Iris by Argon&Co] (https://www.irisbyargonandco.com/fr/)
""")
st.markdown(" ")
st.markdown(" ")

st.markdown('###### 🗃️ Upload Data')
df_path = st.file_uploader("Use the template (link below) to upload a readable dataset", type=["xlsx"])

st.markdown('###### 🧐 Select an Analysis')

# @st.cache(suppress_st_warning=True)
def upload_data():
    df = pd.read_excel(df_path, skiprows=1, sheet_name=0)
    df_carac = 1#pd.read_excel(df_path, sheet_name=1)
    #df = pd.read_excel('Data Files\Data_OVH_2.xlsx', skiprows=1, sheet_name=0)
    #df_carac = pd.read_excel('Data Files\Data_OVH_2.xlsx', sheet_name=1)
    return df, df_carac

if df_path is not None:
    global df2
    df, df_carac = upload_data()
    shared_dataset.input_dataset = df
    shared_dataset.carac_dataset = df_carac

# Add all your application here
app.add_app("Home Page - Upload Data", home.app)
app.add_app("Construct Times Table", Construct_Times_Table.app)
app.add_app("Advanced Statistics", Advanced_Statistics.app)
app.add_app("Find delays causes with subgroups analysis", Causes_Finder_Obs.app)
app.add_app("Find delays causes with bayesian networks (Causal Nex)", Causes_Finder_Causal_Nex.app)
app.add_app("Find delays causes with ML Classification Models", Causes_Finder_ML_Model.app)

# The main app
app.run()
