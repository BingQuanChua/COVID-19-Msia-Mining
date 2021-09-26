import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import math
from PIL import Image


def main():
    st.header("""
        COVID-19 Malaysia, A Data Mining Approach
    """)

    # Import cases and testing data set
    cases_malaysia = pd.read_csv(
        'dataset/cases_and_testing/cases_malaysia.csv')
    cases_state = pd.read_csv('dataset/cases_and_testing/cases_state.csv')
    clusters = pd.read_csv('dataset/cases_and_testing/clusters.csv')
    tests_malaysia = pd.read_csv(
        'dataset/cases_and_testing/tests_malaysia.csv')
    tests_state = pd.read_csv('dataset/cases_and_testing/tests_state.csv')

    # Sidebar
    st.sidebar.header("Table of Content")
    st.sidebar.write("[1. Exploratory Data Analysis](#1-exploratory-data-analysis)")
    st.sidebar.write("[2. Correlation Analysis](#2-correlation-analysis)")
    st.sidebar.write("[3. Strong Features and Indicators](#3-strong-features-and-indicators)")
    st.sidebar.write("[4. Regression and Classification](#4-classification-and-regression)")

    # Introduction
    st.write("""
        This project aims to use data mining techniques to gain some insight from 
        \"[Open Data on COVID-19 in Malaysia](https://github.com/MoH-Malaysia/covid19-public)\" by Ministry of Health (MOH), Malaysia. 
        After detailed analysis, our project will be focusing on data from \"**Cases and Testing**\".
        """)

    # Question 1
    st.write("#Q1")
    st.write("""
        # 1. Exploratory Data Analysis
    """)
    st.write("""Problem Statement: Discuss the exploratory data analysis conducted including detection of outliers and missing values.""")

    st.write("""
        ## 1.1 Select a Dataset
    """)
    datasets = {
        '--- select a dataset ---': 0,
        'cases_malaysia.csv': cases_malaysia,
        'cases_state.csv': cases_state,
        'tests_malaysia.csv': tests_malaysia,
        'tests_state.csv': tests_state,
        'clusters.csv': clusters
    }
    selected_dataset = st.selectbox(
        "Cases and Testing", [key for key in datasets])

    df = datasets.get(selected_dataset)

    # If no data set selected
    if not isinstance(df, pd.DataFrame):
        st.write("#### **Please select a dataset to continue.**")

    # If any data set selected
    if selected_dataset != '--- select a dataset ---':
        st.write("""
            ## 1.2 Dataset
        """)
        st.write(f"Selected Dataset: `{selected_dataset}`")
        st.write(df.head())

        st.write("""
            ## 1.3 Missing value
        """)
        missing_value = df[df.isna().values.any(axis=1)]
        rows = missing_value.shape[0]
        if rows > 0:
            st.write(f"There are {rows} rows with missing values")
            st.write("Identified missing value in each column:")
            st.write(df.isna().sum())
        else:
            st.write("There are no missing values in this dataset")

        st.write("""
            ## 1.4 Outliers
        """)

        # functin declaration
        def plot_boxplot(series, title='', xlabel=''):
            bp = sns.boxplot(x=series)
            bp.set(title=title,
                   xlabel=xlabel)
            return bp

        def check_outlier(df):
            numeric_columns = df.describe().columns.copy()
            num_of_columns = len(numeric_columns)

            for i in range(math.ceil(num_of_columns/2)):
                c = 2*i

                plt.figure(figsize=(20, 2))
                plt.subplot(1, 2, 1)
                plot_boxplot(df[numeric_columns[c]], xlabel=numeric_columns[c])
                plt.subplot(1, 2, 2)
                try:
                    plot_boxplot(df[numeric_columns[c+1]],
                                 xlabel=numeric_columns[c+1])
                except IndexError:
                    plt.xticks([]), plt.yticks([])
                    plt.axis("off")
                st.pyplot(plt)

        # call outlier function
        check_outlier(df)

        # st.write("""
        #     ## 1.5 Other Interesting Exploration
        # """)

    # Question 2
    st.write("# 2 Correlation Analysis")
    st.markdown("Problem Statement: What are the states that exhibit strong correlation with Pahang and Johor?")

    st.markdown("To find the states that correlated to Johor and Pahang, we will use correlation heatmap. If the correlation score between that state and cases_new of Johor or Pahang higher, the correlation between both states are stronger")

    stateQ2 = ["--- select a state ---", "Pahang", "Johor"]

    selected_stateQ3 = st.selectbox("Strong Features", stateQ2)

    if selected_stateQ3 == "Pahang":
        im = Image.open('img/Pahang_state.png')
        st.image(im,width=1000, caption='Heatmap for Pahang with other states')
    elif selected_stateQ3 == "Johor":
        im = Image.open('img/johor_state.png')
        st.image(im,width=1000, caption='Heatmap for Johor with other states')


    # Question 3
    st.write("""
        # 3. Strong Features and Indicators
    """)
    st.write("Problem Statement: What are the strong features to daily cases for Pahang, Kedah,Johor, and Selangor?")

    st.write('For this questions we have to determine featuresa that correlated to daily cases(cases_new) of Pahang, Kedah, Johor and Selangor. To get the results, we will using Correlation Heatmap and Feature Importance method to calculate the correlation score between each variables and target variables(cases_news)')

    st.subheader("Correlation Heatmap")

    st.write("Correlation heatmap is a graphical representation of correlation matrix that representing correlation between different variables.")
    st.write("""According to the heatmaps above, we could determine strong features to cases_new for Penang, Kedah, Johor and Selangor.
    If the value of the correlation score is higher or color of that is brighter, Fthat variable is stronger to cases_new.For i) Penang, variable cases_recovered has highest correlation which is 0.67. 
    For ii) Kedah, there are 3 variables has high correlation to daily cases, which are cases_recovered, rfk_ag and pcr three of them get around 80 of correlation scores. 
    For iii) Johor, we found total 5 variables are highly correlated to daily cases. Five of them are cases_recovered, rtk_ag, pcr, deaths_new and deaths_new_dod. Five of them got 80%% above of correlation scores. 
    Finally, for iv) Selangor, similar like Pahang, each of the variables did not showns very high correlation score to cases_new. The variable deaths_new_dod got the highest correlation score which is 0.62.
    """)

    stateQ3 = ["--- select a state ---", "Pahang", "Kedah", "Johor", "Selangor"]

    selected_stateQ3 = st.selectbox("Heatmap", stateQ3)

    if selected_stateQ3 == "Pahang":
        im = Image.open('img/Pahang_f.png')
        st.image(im, width=700, caption='Heatmap for Pahang')

    elif selected_stateQ3 == "Kedah":
        im = Image.open('img/Kedah_f.png')
        st.image(im, width=700, caption='Heatmap for Kedah')

    elif selected_stateQ3 == "Johor":
        im = Image.open('img/Johor_f.png')
        st.image(im, width=700, caption='Heatmap for Johor')

    elif selected_stateQ3 == "Selangor":
        im = Image.open('img/Selangor_f.png')
        st.image(im, width=700, caption='Heatmap for Selangor')

    st.write('Feature Importance Method')
    st.write('Next, we will using the Feature Importance Method to determine the strong features to daily cases. Feature Importance will assign a score to each of the variables according to how they useful for predicting target variable. If a feature get higher score, which mean it is stronger to daily cases')

    # Question 4
    st.write("# 4. Classification and Regression")

    st.markdown("Problem Statement: Comparing classification and regression model to determine which model perform well in daily cases for Pahang, Kedah, Johor and Selangor")

    st.markdown('For classification, models we will use are K-Nearest Neighbors Classifier, Naive Bayes Clasifier,Decision Tree Classifier and Random Forest Classifier. We divide our target variale to three classes according to binning, which are high, moderate and low daily new cases. For regression, models we will use are Linear Regression, Decision Tree Regressor, Random Forest Regressor and Support Vector Regressor')

    st.write("## 4.1 Classification")

    im = Image.open('img/confusion1.png')  
    st.image(im,width=1000, caption='')

    im = Image.open('img/confusion2.png')  
    st.image(im,width=1000, caption='Confusion Matrix')

    im = Image.open('img/classification1.png')  
    st.image(im,width=1000, caption='')

    im = Image.open('img/classification2.png')  
    st.image(im,width=1000, caption='Evalaution for classification models')

    im = Image.open('img/ROC.png')
    st.image(im, width=800, caption = 'ROC curve')

    st.markdown('For classification model, the evaluation matrix we used are Confusion Matrix, precision, recall, F1-score and ROC curve. According to the evaluation matrix above, it is shown that our best model for classifying the daily cases for Pahang, Kedah, Johor and Selangor is the Random Forest Classifier, following up by KNN and Decision Tree. The Naive Bayes Classifier performs the poorest in this case.')

    st.write("## 4.2 Regression")

    im = Image.open('regression.png')
    st.image(im, width=1200, caption = 'Regression Model Evaluation')

    st.markdown('R Square values ranges from 0 to 1. The higher value of R Square indicates that the model fits the data better. On the other hand, both MAE and RMSE ranges from 0 to  ∞ , and lower values are preferred.In our case, the R Square values of all 4 regressor are very close to 1. However, their MAE and RMSE are very high. We would consider Decision Tree and Random Forest as better regressors in this case as their R Square is higher and the errors are relatively lower than others.')


if __name__ == "__main__":
    main()
