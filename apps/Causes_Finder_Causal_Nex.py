import streamlit as st

import shared_dataset
import sys
sys.path.append('.')
from Stable_Ops_Functions import StabAnalyser, CausalAnalyser, CausalAnalyser_v2


def app():
    time_cols = shared_dataset.time_cols
    all_data_merged = shared_dataset.all_data_merged
    df_carac = shared_dataset.carac_dataset
    possibles_groups = set.union(set(df_carac.columns.to_list()), set(['Start Week Day', 'Start Hour'])) - set(['OF'])

    st.write('### Causes Finder 2 : Causal Model')

    all_data_merged = all_data_merged.fillna(0)
    all_data_merged.to_excel('All_Data_Merged.xlsx')
    causal_analyser = CausalAnalyser(all_data_merged)
    target2 = st.multiselect("Select Target Time to Analyse", time_cols)
    causes_VA = st.multiselect("Possible Causes", possibles_groups)
    tabu_child = st.multiselect("Nodes that are not a consequence", possibles_groups)

    if len(causes_VA) > 1 and (len(target2) > 0):
        st.write("Computing Bayesian Network")
        target2 = target2[0]
        error = True
        edge_value_tresh = 0
        number_of_try = 0
        #while error & (number_of_try < 10):
        try:
            fig, ie, probas = causal_analyser.causes_finder(
                features=causes_VA,
                target=target2, edge_value_tresh=edge_value_tresh,
                tabu_child_nodes=tabu_child)
            error = False
            st.pyplot(fig, height=10)#, use_container_width=True)
            st.table(probas.transpose().reset_index().head())
            error = False
            #except:
            #    edge_value_tresh = edge_value_tresh + 0.05
            #    number_of_try = number_of_try + 1
        except:
        #if error:
            causal_analyser2 = CausalAnalyser_v2(all_data_merged)
            fig, ie, probas = causal_analyser2.causes_finder2(
                features=causes_VA,
                target=target2, edge_value_tresh=edge_value_tresh,
                tabu_child_nodes=tabu_child)



            s2 = probas.transpose().reset_index().head()
            st.pyplot(fig, height=10)#, use_container_width=True)
            st.markdown(" **Conditional Probabilities Table**  ⬇️")
            st.table(s2)
            #st.write('Bayesian Network could not fit the Data')

    st.write(' ')
    st.write('')