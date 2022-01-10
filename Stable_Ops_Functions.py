#!/usr/bin/env python
# coding: utf-8

# In[247]:


import probscale
import pandas as pd
import matplotlib.pyplot as plt
from scipy import stats
import numpy as np
import plotly.figure_factory as ff
import scipy.stats as stats
import plotly.express as px
import causalnex
from causalnex.structure.notears import from_pandas
import warnings
warnings.filterwarnings("ignore")
from sklearn.preprocessing import LabelEncoder
from causalnex.discretiser import Discretiser
from causalnex.network import BayesianNetwork
from causalnex.discretiser.discretiser_strategy import DecisionTreeSupervisedDiscretiserMethod
from causalnex.inference import InferenceEngine
from causalnex.structure import DAGRegressor

from sklearn.model_selection import cross_val_score
from sklearn.model_selection import KFold
from itertools import combinations



#from causalnex.plots import plot_structure, NODE_STYLE, EDGE_STYLE
from sklearn.datasets import load_diabetes
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split
import networkx as nx




class StabAnalyser():

    def __init__(self, data):
        self.data = data


    def group_analyser(self, group_col,val_col, high_limit, bin_size = 0.1):

        data_to_use = self.data.copy()
        data_to_use = data_to_use[(data_to_use[val_col] < high_limit) & (data_to_use[val_col] > 0)]


        if group_col == group_col:
            data_to_use = data_to_use[~data_to_use[group_col].isna()]
            data_to_use[group_col] = data_to_use[group_col].apply(str)
            tmp = data_to_use.groupby(group_col).agg({val_col:list})
            tmp = tmp.reset_index()
            tmp['len'] = tmp[val_col].apply(len)
            tmp = tmp[tmp['len'] > 10]
            hist_data = tmp[val_col].values.tolist()
            group_labels = tmp[group_col].values.tolist() # name of the datasets
            #try:
            fig1 = ff.create_distplot(hist_data, group_labels, bin_size=bin_size)
            print('ok')
            fig2 = px.box(data_to_use, x=val_col, y=group_col, color = group_col,
                          color_discrete_sequence=px.colors.qualitative.G10).update_yaxes(
                categoryorder="category descending")  # , color = group_col)

        else:
            s = data_to_use[val_col].replace(0, np.nan).dropna()
            hist_data = [s]
            group_labels = ['Distplot']  # name of the dataset

            bin_size = (s.max() - s.min()) / 100

            fig1 = ff.create_distplot(hist_data, group_labels, bin_size=bin_size)
            fig2 = px.box(data_to_use, x=val_col)

        return fig1, fig2,hist_data,group_labels

    def group_scatter(self, group_col, val_col, y_axis):
        fig = px.scatter(self.data, x=val_col, y=y_axis, facet_col=group_col, trendline="ols")
        fig.show()

    def mean_hypothesis_test(self, group_col, val_col, high_limit):
        data_to_use = self.data.copy()
        data_to_use = data_to_use[(data_to_use[val_col] < high_limit) & (data_to_use[val_col] > 0)]

        if group_col == group_col:
            data_to_use = data_to_use[~data_to_use[group_col].isna()]
            data_to_use[group_col] = data_to_use[group_col].apply(str)
            tmp = data_to_use.groupby(group_col).agg({val_col: list})
            tmp = tmp.reset_index()
            tmp['len'] = tmp[val_col].apply(len)
            tmp = tmp[tmp['len'] > 10]
            hist_data = tmp[val_col].values.tolist()
            group_labels = tmp[group_col].values.tolist()  # name of the datasets

        def ttest_run(c1, c2, name1, name2):
            results = stats.ttest_ind(c1, c2, equal_var=False)
            df = pd.DataFrame({'Group 1': name1,
                               'Group 2': name2,
                               't-stat': round(results.statistic,3),
                               'p-value': round(results.pvalue,3), 'Ho Rejected': results.pvalue < 0.05},
                              index=[0])
            return df

        df_list = [ttest_run(tmp.loc[i][val_col], tmp.loc[j][val_col],
                             tmp.loc[i][group_col], tmp.loc[j][group_col]) for i, j in combinations(tmp.index, 2)]
        final_df = pd.concat(df_list, ignore_index=True)
        return final_df

    def histogram_plot(self, val_col):

        s = self.data[val_col].replace(0, np.nan).dropna()
        hist_data = [s]
        group_labels = ['Distplot']  # name of the dataset

        bin_size = (s.max() - s.min())/100

        fig = ff.create_distplot(hist_data, group_labels, bin_size=bin_size)
        return fig

    def density_plots(self, val_col):
        s = self.data[val_col].apply(lambda x: max(x, 0)).replace(0, np.nan).dropna()

        common_opts = dict(probax='y', datalabel='Serie', scatter_kws=dict(marker='.', linestyle='none'))

        fig, ((ax1, ax2), (ax3, ax4)) = plt.subplots(figsize=(20, 10), nrows=2, ncols=2)

        s.plot.density(ax=ax1)

        probscale.probplot(s, plottype='pp', ax=ax2,
                                 problabel='Percentiles', **common_opts)
        probscale.probplot(s, plottype='prob', ax=ax3,
                                 problabel='Normal Probabilities', **common_opts)
        probscale.probplot(s, plottype='prob', ax=ax4,
                                 problabel='Normal Probabilities', probax='y', datascale='log',
                                 datalabel='Log Task Time', scatter_kws=dict(marker='.', linestyle='none'),
                                 bestfit=True, estimate_ci=True,
                                 line_kws={'label': 'BF Line', 'color': 'b'})
        return fig

    def fit_stat_law(self, val_col, law, **params_law):
        s = self.data[val_col].apply(lambda x: max(x, 0)).replace(0, np.nan).dropna()
        dist = law
        dist_params = dist.fit(s, **params_law)
        dist_ = dist(*dist_params)

        y, x = np.histogram(s, bins=200, density=True)
        x = (x + np.roll(x, -1))[:-1] / 2.0

        arg = dist_params[:-2]
        loc = dist_params[-2]
        scale = dist_params[-1]

        # Calculate fitted PDF and error with fit in distribution
        p = dist.pdf(x, loc=loc, scale=scale, *arg)
        squared_estimate_errors = np.sum(np.power(y - p, 2.0))
        print(f'RMSE {squared_estimate_errors}')

        ### AIC Compute
        logLik = np.sum(dist_.logpdf(s))
        k = len(dist_params)
        aic = 2 * k - 2 * (logLik)

        print(f'AIC {aic}')

        print(' ')
        fig = plt.figure()
        plt.plot(x, p)
        plt.plot(x, y)
        plt.title('Density Fit')



        fig2, ax2 = plt.subplots(figsize=(10, 7))

        common_opts = dict(probax='y', datalabel='Serie', scatter_kws=dict(marker='.', linestyle='none'))
        probscale.probplot(s, plottype='pp', ax=ax2,
                           problabel='Percentiles', **common_opts)

        x = np.linspace(s.min(),s.max())
        ax2.plot(x, dist_.cdf(x) * 100,
                'r-', lw=5, alpha=0.6, label='norm pdf')

        plt.title('Probability Plot Fit')
        #plt.show()
        return fig, fig2, squared_estimate_errors, aic, dist_params,dist_


class CausalAnalyser():

    def __init__(self, data0):
        self.data = data0

    def causes_finder(self, features, target, edge_value_tresh = 0, tabu_child_nodes=None):

        tmp0 = self.data
        tmp0.columns = [i.replace(' ', '_').replace('é', 'e').replace('-', '_') for i in tmp0.columns]  ##
        features = [i.replace(' ', '_').replace('é', 'e').replace('-', '_') for i in features]  ##
        target = target.replace(' ', '_').replace('é', 'e').replace('-', '_')
        tabu_child_nodes = [i.replace(' ', '_').replace('é', 'e').replace('-', '_') for i in tabu_child_nodes]  ##

        X, y = tmp0[features], tmp0[target]

        data = tmp0[features + [target]].rename(columns={target: 'target'})


        ## Encode Categorcial for Structure Learning
        struct_data = data.copy()
        non_numeric_columns = list(struct_data.select_dtypes(exclude=[np.number]).columns)
        le = LabelEncoder()
        for col in non_numeric_columns:
            struct_data[col] = le.fit_transform(struct_data[col])

        ## Learn Structure
        sm = from_pandas(struct_data, tabu_parent_nodes=['target'], tabu_child_nodes=tabu_child_nodes)
        sm.remove_edges_below_threshold(edge_value_tresh)
        fig, ax = plt.subplots(figsize=(20, 10))
        nx.draw_networkx(sm, ax=ax)
        fig.show()

        sm = sm.get_largest_subgraph()

        ## Discretize for Probabilities Learning
        tree_discretiser = DecisionTreeSupervisedDiscretiserMethod(mode="single", tree_params={"max_depth": 2,
                                                                                               "random_state": 2021})


        tresh = 10  # Treshold of number of disctinct value for treshold
        features_to_discret = list(self.data[features].nunique().to_frame().rename({0: 'v'},
                                                                                   axis=1).query('v > @tresh').index)

        data[features_to_discret] = data[features_to_discret].applymap(float)

        tree_discretiser.fit(
            feat_names=features_to_discret,
            dataframe=data[features_to_discret + ['target']],
            target_continuous=True,
            target="target",
        )

        data_for_proba = data.copy()
        for col in features:
            if col in features_to_discret:
                data_for_proba[col] = tree_discretiser.transform(data[[col]])

        # Discretize Target

        #nb_target_group = 20
        #splits = list(np.linspace(data_for_proba["target"].min(), data_for_proba["target"].max(), nb_target_group))
        #data_for_proba["target"] = Discretiser(method="fixed",
        #                                       numeric_split_points=splits).transform(data["target"])

        quantile_1 = 0.9
        quantile_2 = 0.75

        q1 = data_for_proba["target"].quantile(quantile_1)
        q2 = data_for_proba["target"].quantile(quantile_2)

        def f(x):
            if x <= q1:
                return 'Normal'
            elif x <= q2:
                return 'Quantile 0.75 - 0.9'
            else:
                return 'Quantile 0.9 - 1'

        data_for_proba["target"] = data_for_proba["target"].apply(f)
        map_disc1 = tree_discretiser.map_thresholds

        bn = BayesianNetwork(sm)

        bn = bn.fit_node_states(data_for_proba)
        bn = bn.fit_cpds(data_for_proba, method="BayesianEstimator", bayes_prior="K2")

        print('Conditional Probabilities')
        probas_table = bn.cpds['target']

        ie = InferenceEngine(bn)

        ## Check Overfitting
        train, test = train_test_split(data_for_proba, train_size=0.9, test_size=0.1, random_state=7)


        return fig, ie, probas_table


class CausalAnalyser_v2():

    def __init__(self, data0):
        self.data = data0


    def causes_finder2(self, features, target, edge_value_tresh = 0, tabu_child_nodes=None):

        ### Data has to be continuous

        reg = DAGRegressor(
            alpha=0.1,
            beta=0.9,
            hidden_layer_units=None,
            dependent_target=True,
            enforce_dag=True,
        )

        tmp0 = self.data
        tmp0.columns = [i.replace(' ', '_').replace('é', 'e') for i in tmp0.columns]  ##
        features = [i.replace(' ', '_').replace('é', 'e') for i in features]  ##
        target = target.replace(' ', '_').replace('é', 'e')
        #tabu_child_nodes = [i.replace(' ', '_').replace('é', 'e') for i in tabu_child_nodes]  ##

        #X, y = tmp0[features], tmp0[target]

        data = tmp0[features + [target]].rename(columns={target: 'target'})

        ## Encode Categorical for Structure Learning with DAG Regressor
        struct_data = data.copy()
        non_numeric_columns = list(struct_data.select_dtypes(exclude=[np.number]).columns)
        le = LabelEncoder()
        for col in non_numeric_columns:
            struct_data[col] = le.fit_transform(struct_data[col])


        scores = cross_val_score(reg, struct_data[features], struct_data['target'],
                                 cv=KFold(shuffle=True, random_state=42))
        print(f'MEAN R2: {np.mean(scores).mean():.3f}')

        X = struct_data[features]
        y = struct_data['target']
        reg.fit(X, y)
        print(pd.Series(reg.coef_, index=features))
        graph = reg.plot_dag()
        fig, ax = plt.subplots()
        nx.draw_networkx(graph, ax=ax)


        ## Discretize for Probabilities Learning
        tree_discretiser = DecisionTreeSupervisedDiscretiserMethod(mode="single", tree_params={"max_depth": 3,
                                                                                               "random_state": 2021})

        tresh = 10  # Treshold of number of disctinct value for treshold
        features_to_discret = list(self.data[features].nunique().to_frame().rename({0: 'v'},
                                                                                   axis=1).query('v > @tresh').index)

        data[features_to_discret] = data[features_to_discret].applymap(float)

        tree_discretiser.fit(
            feat_names=features_to_discret,
            dataframe=data[features_to_discret + ['target']],
            target_continuous=True,
            target="target",
        )

        data_for_proba = data.copy()

        for col in features:
            if col in features_to_discret:
                #print(data_for_proba[col].head())
                data_for_proba[col] = tree_discretiser.transform(data[[col]])
                #print(data_for_proba[col].head())

        def bound_name(x, bound):
            a = bound[x]
            b = bound[x+1]
            return f"g{x} ( {a} - {b} )"

        for col in features_to_discret:

            bounds = list(tree_discretiser.map_thresholds[col])
            bounds.insert(0, 0)
            bounds.append(np.inf)

            data_for_proba[col] = data_for_proba[col].apply(lambda x : bound_name(x, bounds))

        # Discretize Target

        #nb_target_group = 20
        #splits = list(np.linspace(data_for_proba["target"].min(), data_for_proba["target"].max(), nb_target_group))
        #data_for_proba["target"] = Discretiser(method="fixed",
        #                                       numeric_split_points=splits).transform(data["target"])

        quantile_1 = 0.7
        quantile_2 = 0.9


        q1 = round(data_for_proba["target"].quantile(quantile_1), 1)
        q2 = round(data_for_proba["target"].quantile(quantile_2), 1)

        def f(x):
            if x <= q1:
                return f'Quantile 0 - 0.7 ( 0 - {q1} )'
            elif x <= q2:
                return f'Quantile 0.7 - 0.9 ( {q1} - {q2} )'
            else:
                return f'Quantile 0.9 - 1 ( {q2} - inf )'

        data_for_proba["target"] = data_for_proba["target"].apply(f)

        bn = BayesianNetwork(graph)
        bn = bn.fit_node_states(data_for_proba)
        # train = train.applymap(str)
        bn = bn.fit_cpds(data_for_proba, method="BayesianEstimator", bayes_prior="K2")
        ie = InferenceEngine(bn)

        probas_table = bn.cpds['target'].transpose().sort_index()
        probas_table_rescale = probas_table.copy()
        probas_table_rescale[f'Quantile 0 - 0.7 ( 0 - {q1} )'] = probas_table_rescale[f'Quantile 0 - 0.7 ( 0 - {q1} )'] / 0.7
        probas_table_rescale[f'Quantile 0.7 - 0.9 ( {q1} - {q2} )'] = probas_table_rescale[
                                                                     f'Quantile 0.7 - 0.9 ( {q1} - {q2} )'] / 0.2
        probas_table_rescale[f'Quantile 0.9 - 1 ( {q2} - inf )'] = probas_table_rescale[
                                                                          f'Quantile 0.9 - 1 ( {q2} - inf )'] / 0.1

        return fig, ie, probas_table, probas_table_rescale
