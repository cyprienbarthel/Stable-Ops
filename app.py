import streamlit as st
from multiapp import MultiApp
from apps import home, Ordered_Production_Data_Analysis # import your app modules here


app = MultiApp()
col1, col2, col3 = st.columns(3)
with col1:
    st.image('Data Files\IRIS_LOGO.PNG')
with col3:
    st.image('Data Files\OVH_Logo.PNG')

st.markdown("""
### Operations Stability Analysis
###### Data Visualization App made by [@Iris by Argon&Co] (https://www.irisbyargonandco.com/fr/)
""")
st.markdown(" ")
st.markdown(" ")

# Add all your application here
app.add_app("Default", home.app)
app.add_app("Supposedly Ordered Production Data", Ordered_Production_Data_Analysis.app)

# The main app
app.run()
