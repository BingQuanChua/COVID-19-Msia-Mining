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

        st.write("""
            ## 1.5 Other Interesting Exploration
        """)

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

    state = ["--- select a state ---", "Pahang", "Kedah", "Johor", "Selangor"]

    selected_state = st.selectbox("Cases and Testing", state)

    if selected_state == "Pahang":
        im = Image.open('img/Pahang_f.png')
        st.image(im, width=700, caption='Heatmap for Pahang')

    elif selected_state == "Kedah":
        im = Image.open('img/Kedah_f.png')
        st.image(im, width=700, caption='Heatmap for Kedah')

    elif selected_state == "Johor":
        im = Image.open('img/Johor_f.png')
        st.image(im, width=700, caption='Heatmap for Johor')

    elif selected_state == "Selangor":
        im = Image.open('img/Selangor_f.png')
        st.image(im, width=700, caption='Heatmap for Selangor')

    st.write('Feature Importance Method')
    st.write('Next, we will using the Feature Importance Method to determine the strong features to daily cases. Feature Importance will assign a score\nto each of the variables according to how they useful for predicting target variable.\n If a feature get higher score, which mean it is stronger to daily cases')


if __name__ == "__main__":
    main()
