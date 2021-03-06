import streamlit as st
import shared_dataset
import seaborn as sns
import sys
sys.path.append('.')
from Stable_Ops_Functions import StabAnalyser, CausalAnalyser, CausalAnalyser_v2


def app():
    time_cols = shared_dataset.time_cols
    all_data_merged = shared_dataset.all_data_merged
    df_carac = shared_dataset.carac_dataset

    st.write('### Causes Finder 1 : Analyse Subgroups Statistics')

    Stab_analyser = StabAnalyser(all_data_merged)
    target = st.multiselect("Select Target Time to analyse", time_cols)
    possibles_groups = set.union(set(df_carac.columns.to_list()),set(['Start Week Day','Start Hour']))- set(['OF'])
    group_variable = st.multiselect("Select Grouping Column to analyse statistical differences", possibles_groups)
    int_val = st.slider('Max value of histograms', min_value=0, max_value=100, value=30, step=2)
    bin_size = st.slider('Bin size of density Histogram', min_value=0.1, max_value=5.0, value=0.2, step=0.1)
    if (len(group_variable) > 0) and  (len(target) > 0):
        group_variable = group_variable[0]
        target = target[0]
        st.write(group_variable)
        fig1, fig2, hist, group = Stab_analyser.group_analyser(group_variable, target, int_val, bin_size)
        st.plotly_chart(fig2, use_container_width=True)
        st.plotly_chart(fig1, use_container_width=True)

        st.markdown('##### T-test on whether sub groups have the same mean ')
        st.markdown('We test the null hypothesis **Ho** : *The two subgroups have the same mean*')
        st.markdown("If the p-value of the t-test is < 0.05, we reject Ho with a risk of failure of 5% \
                    Else, we don't reject Ho (but we cannot say if Ho is true)")

        df_tests = Stab_analyser.mean_hypothesis_test(group_variable, target, int_val)

        cm = sns.light_palette("green", as_cmap=True)
        st.dataframe(df_tests.style.background_gradient(cmap=cm, low=.5, high=0))#.highlight_null('red'))
    else:
        pass