import streamlit as st
import shared_dataset
import sys
sys.path.append('.')
from sklearn.tree import DecisionTreeClassifier
from sklearn.preprocessing import LabelEncoder
from sklearn.model_selection import train_test_split
import matplotlib.pyplot as plt
from sklearn import tree
from sklearn.metrics import *
import pandas as pd
from numpy import argmax
import seaborn as sns
import numpy as np
import plotly.express as px
from imblearn.over_sampling import SMOTE, ADASYN
from bayes_opt import BayesianOptimization
import xgboost as xgb
from xgboost import XGBClassifier
from io import BytesIO


def plot_confusion_matrix(y_true, y_pred, class_names, normalize=None,
                          title='Confusion Matrix', plot_numbers=True, display_names=None,
                          figsize=(5, 5)):

    cm = confusion_matrix(y_true, y_pred, labels=class_names)

    if not display_names:
        display_names = class_names
    df_cm = pd.DataFrame(cm, index=display_names, columns=display_names)
    fig = plt.figure(figsize=figsize)
    sns.heatmap(df_cm, annot=plot_numbers, cmap='Blues', fmt='g')
    plt.setp(plt.gca().get_xticklabels(), ha="right", rotation=45)
    plt.ylabel('True Label')
    plt.xlabel('Predicted Label')
    #plt.title(title)
    return fig

def app():
    time_cols = shared_dataset.time_cols
    all_data_merged = shared_dataset.all_data_merged
    df_carac = shared_dataset.carac_dataset
    possibles_groups = set.union(set(df_carac.columns.to_list()), set(['Start Week Day', 'Start Hour'])) - set(['OF'])
    st.write('### Causes Finder 3 : Machine Learning Model for Detecting Outliers')
    df = all_data_merged.fillna(0)

    target = st.sidebar.multiselect("Select Target Time to Analyse", time_cols)
    quantile = st.sidebar.slider("Select the quantile that defines an outlier", min_value=0.75, max_value=0.99, value=0.9,
                         step=0.01)
    choices = list(set.union({'- Select All'}, possibles_groups))
    features = st.sidebar.multiselect("Possible Causes (features of ML model)", choices)

    model = st.sidebar.radio('Select a Model', ['Decision Tree', 'XG Boost'])

    if (len(features) > 0) and (len(target) > 0):
        target = target[0]
        if (features[0] == '- Select All') :
            features = list(possibles_groups)

        q = df[target].quantile(quantile)

        X = df[features]
        y = df[target] > q
        non_numeric_columns = list(X.select_dtypes(exclude=[np.number]).columns)
        le = LabelEncoder()
        for col in non_numeric_columns:
            X[col] = le.fit_transform(X[col])

        y = le.fit_transform(y)
        X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.30, random_state=2)

        resample = st.sidebar.radio('Resampling (with ASADYN)', ('Yes','No'))
        if resample == 'Yes':
            X_train, y_train = ADASYN().fit_resample(X_train, y_train)

        if model == 'Decision Tree':
            classifier = DecisionTreeClassifier(max_depth=4, min_samples_leaf=10, random_state=1)
            classifier.fit(X_train, y_train)
            y_pred = classifier.predict(X_test)

            fig2 = tree.export_graphviz(classifier, out_file=None, filled=True, feature_names=X.columns, rotate=True)
            st.write('???? Decision Tree ')
            st.graphviz_chart(fig2)


        if model == 'XG Boost':

            # Converting the dataframe into XGBoost???s Dmatrix object
            dtrain = xgb.DMatrix(X_train, label = y_train)

            # Bayesian Optimization function for xgboost
            # specify the parameters you want to tune as keyword arguments
            def bo_tune_xgb(max_depth, gamma, n_estimators, learning_rate):
                params = {'max_depth': int(max_depth),
                          'gamma': gamma,
                          'n_estimators': int(n_estimators),
                          'learning_rate': learning_rate,
                          'subsample': 0.8,
                          'eta': 0.1,
                          'eval_metric': 'rmse'}
                # Cross validating with the specified parameters in 5 folds and 70 iterations
                cv_result = xgb.cv(params, dtrain, num_boost_round=70, nfold=5, metrics="auc", stratified=True)
                # Return the negative RMSE
                return -1.0 * cv_result['test-auc-mean'].iloc[-1]

            # Invoking the Bayesian Optimizer with the specified parameters to tune
            xgb_bo = BayesianOptimization(bo_tune_xgb, {'max_depth': (3, 5),
                                                        'gamma': (0.05, 1),
                                                        'learning_rate': (0.05, 0.8),
                                                        'n_estimators': (100, 120)
                                                        })

            # performing Bayesian optimization for 5 iterations with 8 steps of random exploration with an #acquisition function of expected improvement
            xgb_bo.maximize(n_iter=5, init_points=8, acq='ei')
            # Extracting the best parameters
            params = xgb_bo.max['params']
            # Converting the max_depth and n_estimator values from float to int
            params['max_depth'] = int(params['max_depth'])
            params['n_estimators'] = int(params['n_estimators'])
            st.write('** Best Params : **')
            st.write(params)

            ### RMSE Curve with best params
            st.write('Loss Curve among Iterations')
            cv_results = xgb.cv(params, dtrain, num_boost_round=70, nfold=5, metrics=["rmse", "auc"])
            width = st.sidebar.slider("plot width", 1, 25, 3)
            height = st.sidebar.slider("plot height", 1, 25, 1)
            fig, ax = plt.subplots(figsize=(width, height))
            ax.plot(cv_results['train-rmse-mean'])
            ax.plot(cv_results['test-rmse-mean'])
            buf = BytesIO()
            fig.savefig(buf, format="png")
            st.image(buf)

            ## Predict
            classifier = XGBClassifier(**params, objective='binary:logistic', eval_metric="auc").fit(X_train, y_train)
            y_pred = classifier.predict(X_test)
            # Looking at the classification report
            #st.write(classification_report(y_pred_xgb, y_test))

        st.write(" ???? Features Importance")
        feat_importances = pd.DataFrame(classifier.feature_importances_, index=X_test.columns,
                                        columns=['Feature Importance'])

        fig2 = px.bar(feat_importances.reset_index(), x='index', y='Feature Importance',
                      color='Feature Importance', height=400).update_xaxes(categoryorder="total descending")
        st.plotly_chart(fig2, use_container_width=True)

        st.markdown(" ?????? **Confusion Matrix and Classification Metrics**")
        fig = plt.figure()
        fig = plot_confusion_matrix(y_test, y_pred, [0, 1])
        y_probas = classifier.predict_proba(X_test)[:, 1]
        precision, recall, thresholds = precision_recall_curve(y_test, y_probas)

        fig1 = plt.figure()
        plt.plot(recall, precision, marker='.')
        plt.xlabel('Recall')
        plt.ylabel('Precision')


        col1, col2 = st.columns(2)
        with col1:
            st.pyplot(fig, use_container_width=True)
        with col2:
            st.pyplot(fig1, use_container_width=True)

        precision, recall, thresholds = precision[1:-1], recall[1:-1], thresholds[1:-1]
        st.markdown(" **Changing Probability Score**")
        score = 2*(precision * recall) / (precision + recall)  ## Score pour trouver le meilleur compromis
        # locate the index of the largest  score
        ix = argmax(score) - 1
        st.markdown('Best Threshold=%f, Score=%.3f' % (thresholds[ix], score[ix]))
        y_pred2 = [1 if x >= thresholds[ix] else 0 for x in list(y_probas)]

        col1, col2 = st.columns(2)
        with col1:
            fig = plot_confusion_matrix(y_test, y_pred2, class_names=[0, 1])
            st.pyplot(fig)
        with col2:
            st.markdown("##### New Metrics")
            st.markdown(f"Balanced Accuracy : {balanced_accuracy_score(y_test, y_pred2)}")
            st.markdown(f"Precision : {precision_score(y_test, y_pred2)}")
            st.markdown(f"Recall : {recall_score(y_test, y_pred2)}")


