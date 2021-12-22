import streamlit as st
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import re
import pandas as pd
import random
import seaborn as sns

def app():
    st.title('Correlation Analysis')

    data_path = 'C:/Users/CYB/OneDrive - Argon Consulting/Chloé - Early Signals - Code/Tables Power BI/'

    @st.cache
    def load_data(data_path: str):
        return pd.read_csv(data_path + "sales_all.csv")

    st.text(
        "Affichage des courbes des ventes cumulées d'une maille donnée \n(worldwide, une entité, un pays) vs les ventes "
        "cumulées des différentes boutiques \ncontenus dans cette maille")
    transactions_light = load_data(data_path)
    transactions_light = transactions_light[transactions_light["Week rebased with w0 line region"] <= 36]
    # Select the line

    line = str(st.sidebar.selectbox("Select the line:",
                                    options=sorted(map(str, transactions_light["F&A Commercial Line"].unique()))))
    transactions_flgshp_light = transactions_light[transactions_light["Boutique Type _F&A_"] == "Flagship"]

    # Select the level
    st.sidebar.text("Different levels :\n0 : Worldwide,\n1 : Entity (Region),\n2 : Country")
    level = int(st.sidebar.selectbox("Select the level key", options=[0, 1, 2]))

    # Select the level_name
    if level == 0:
        level_name = "Worldwide"
    else:
        if level == 1:
            level_name = str(st.sidebar.selectbox("Which entity?", sorted(map(str, list(transactions_light.loc[
                                                                                            transactions_light[
                                                                                                "F&A Commercial Line"] == line, "Reporting Entity_Hier"].unique())))))
        if level == 2:
            level_name = str(st.sidebar.selectbox("Which country?", sorted(map(str, list(transactions_light.loc[
                                                                                             transactions_light[
                                                                                                 "F&A Commercial Line"] == line, "Country_boutique"].unique())))))

    # Lags definition

    def colors_gen(n):  # to generate different colors for the further plots
        ret = []
        r = int(random.random() * 256)
        g = int(random.random() * 256)
        b = int(random.random() * 256)
        step = 256 / n
        for i in range(n):
            r += step
            g += step
            b += step
            r = int(r) % 200
            g = int(g) % 256
            b = int(b) % 256
            ret.append((r / 255, g / 255, b / 255))
        return ret

    def double_scale_plot(line_cumsum_hor: pd.DataFrame, lag: int, line: str, rebasement: str, level="Worldwide"):
        fig, ax1 = plt.subplots(figsize=(10, 6))

        ax1.set_xlabel(rebasement)
        ax1.set_ylabel(level + ' sales', color="red")
        ax1.plot(line_cumsum_hor.index, line_cumsum_hor[level], color="red", label=level)
        ax1.tick_params(axis='y', labelcolor="red")
        ax1.legend(loc="upper center")

        ax2 = ax1.twinx()  # instantiate a second axes that shares the same x-axis

        boutique_col = list(line_cumsum_hor.columns)
        boutique_col.remove(level)
        colors = colors_gen(len(boutique_col))  # generate different colors for the different plots

        ax2.set_ylabel('Boutique_sales', color=colors[0])  # we already handled the x-label with ax1
        ax2.tick_params(axis='y', labelcolor=colors[0])
        for boutique_name in boutique_col:
            ax2.plot(line_cumsum_hor.index, line_cumsum_hor[boutique_name], label=boutique_name, color=colors.pop())
        ax2.legend()

        fig.tight_layout()  # otherwise the right y-label is slightly clipped
        fig.suptitle('line: ' + line + ',lag ' + str(lag), x=0.2, y=0.5)

        return fig;

    if level == 0:
        level_transaction = transactions_light
        rebasement = "Week rebased with w0 line"
    if level == 1:
        level_transaction = transactions_light[transactions_light["Reporting Entity_Hier"] == level_name]
        rebasement = "Week rebased with w0 line region"
    if level == 2:
        level_transaction = transactions_light[transactions_light["Country_boutique"] == level_name]
        rebasement = "Week rebased with w0 line region"

    line_transactions = level_transaction[level_transaction["F&A Commercial Line"] == line]
    multiselect_options = list(line_transactions["Boutique Type _F&A_"].unique())
    boutique_types = st.sidebar.multiselect(label="Select the different types of boutiques to plot",
                                            options=multiselect_options, default=multiselect_options[0])

    line_transac_boutique = line_transactions.loc[
                            line_transactions["Boutique Type _F&A_"].apply(lambda boutique: boutique in boutique_types),
                            :]

    # qty sold sum by flgshp by week
    line_boutique_sum = (line_transac_boutique.loc[:, ["boutique_name", "Sales Quantity", rebasement]]
                         .groupby(by=["boutique_name", rebasement])
                         .sum())
    # cumsum by flgshp by week
    line_boutique_cumsum = line_boutique_sum.groupby("boutique_name").cumsum()
    line_boutique_cumsum = line_boutique_cumsum.reset_index()

    # we want to have flgshp name as columns
    line_boutique_cumsum_hor = pd.crosstab(line_boutique_cumsum[rebasement],
                                           line_boutique_cumsum["boutique_name"],
                                           values=line_boutique_cumsum["Sales Quantity"],
                                           aggfunc='mean')

    # level sales cumsum
    level_sales_sum = (line_transactions.loc[:, ["Sales Quantity", rebasement]]
                       .groupby(by=[rebasement])
                       .sum())
    level_sales_cumsum = level_sales_sum.cumsum()

    # merging the both
    line_cumsum_hor = line_boutique_cumsum_hor.merge(level_sales_cumsum, on=rebasement, how="left")
    line_cumsum_hor = line_cumsum_hor.rename(columns={"Sales Quantity": level_name})

    # filling null values
    null_loc = np.where(line_cumsum_hor.isnull())
    null_loc_abs = null_loc[0]
    null_loc_ord = null_loc[1]
    for i in range(len(null_loc_abs)):
        if null_loc_abs[i] == 0:
            line_cumsum_hor.iloc[null_loc_abs[i], null_loc_ord[i]] = 0
        else:
            line_cumsum_hor.iloc[null_loc_abs[i], null_loc_ord[i]] = line_cumsum_hor.iloc[
                null_loc_abs[i] - 1, null_loc_ord[i]]

    # adding lag to data
    global_corr = pd.DataFrame({"type_boutique": [
        line_transac_boutique.loc[line_transac_boutique["boutique_name"] == boutique, "Boutique Type _F&A_"].unique()[0]
        for boutique in line_boutique_cumsum_hor.columns]})
    for lag in [0, 1, 2, 3, 4]:
        initial_length = line_cumsum_hor.shape[0]
        line_cumsum_hor_lag = pd.DataFrame([[np.nan] * line_cumsum_hor.shape[1]] * lag,
                                           columns=line_cumsum_hor.columns).append(line_cumsum_hor, ignore_index=True)
        line_cumsum_hor_lag.loc[:initial_length - 1, level_name] = list(
            line_cumsum_hor_lag.loc[line_cumsum_hor_lag.shape[0] - initial_length:, level_name])
        line_cumsum_hor_lag.loc[initial_length:, level_name] = np.nan

        correlation = pd.DataFrame({"Correlation with " + level_name: [
            line_cumsum_hor_lag[level_name].astype(float).corr(line_cumsum_hor_lag[boutique].astype(float))
            for boutique in line_boutique_cumsum_hor.columns]},
                                   index=line_boutique_cumsum_hor.columns)

        st.pyplot(double_scale_plot(line_cumsum_hor_lag, lag, line, rebasement, level_name))
        global_corr["lag" + str(lag)] = list(correlation["Correlation with " + level_name])

    global_corr.index = list(correlation.index)
    st.text("Correlations entre les différentes boutiques laggés et la grande maille")
    cm = sns.light_palette("green", as_cmap=True)
    st.dataframe(global_corr.style.background_gradient(cmap=cm, low=.5, high=0).highlight_null('red'))

