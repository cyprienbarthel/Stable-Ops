import streamlit as st
from multiapp import MultiApp
from apps import home, Ordered_Production_Data_Analysis # import your app modules here
import pandas as pd
import shared_dataset

app = MultiApp()
#col1, col2, col3 = st.columns(3)
#with col1:
#    st.image('Data Files\IRIS_LOGO.PNG')
#with col3:
#    st.image('Data Files\OVH_Logo.PNG')

st.markdown("""
## Operations Stability Analysis
###### 🚀 Data Visualization App made by [@Iris by Argon&Co] (https://www.irisbyargonandco.com/fr/)
""")
st.markdown(" ")
st.markdown(" ")

st.markdown('###### 🗃️ Upload Data')
df_path = st.file_uploader("Use the template (link below) to upload a readable dataset", type=["xlsx"])

st.markdown('###### 🧐 Select an Analysis')

# @st.cache(suppress_st_warning=True)
def upload_data():
    #df = pd.read_excel(df_path, skiprows=1,sheet_name=0)
    df = pd.read_excel('Data Files\Data_OVH_2.xlsx', skiprows=1, sheet_name=0)
    return df

if df_path is None:
    global df2
    df2 = upload_data()
    shared_dataset.input_dataset = df2

# Add all your application here
app.add_app("Home Page - Upload Data", home.app)
app.add_app("Supposedly Ordered Production Data", Ordered_Production_Data_Analysis.app)

# The main app
app.run()
