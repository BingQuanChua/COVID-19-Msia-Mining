import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import math
from PIL import Image


def main():

    st.set_page_config(layout="wide")

    st.title("""
        COVID-19 Malaysia, A Data Mining Approach
    """)

    # Import cases and testing data set
    cases_malaysia = pd.read_csv('dataset/cases_and_testing/cases_malaysia.csv')
    cases_state = pd.read_csv('dataset/cases_and_testing/cases_state.csv')
    clusters = pd.read_csv('dataset/cases_and_testing/clusters.csv')
    tests_malaysia = pd.read_csv('dataset/cases_and_testing/tests_malaysia.csv')
    tests_state = pd.read_csv('dataset/cases_and_testing/tests_state.csv')
    deaths_malaysia = pd.read_csv('dataset/deaths/deaths_malaysia.csv')
    deaths_state = pd.read_csv('dataset/deaths/deaths_state.csv')

    # Sidebar
    st.sidebar.header("Table of Content")
    st.sidebar.write("[1. Exploratory Data Analysis](#1-exploratory-data-analysis)")
    st.sidebar.write("[2. Correlation Analysis](#2-correlation-analysis)")
    st.sidebar.write("[3. Strong Features and Indicators](#3-strong-features-and-indicators)")
    st.sidebar.write("[4. Regression and Classification](#4-classification-and-regression)")

    # Introduction
    st.write("""
        This project aims to use data mining techniques to gain some insight from 
        \"[Open Data on COVID-19 in Malaysia](https://github.com/MoH-Malaysia/covid19-public)\" by the Ministry of Health (MOH), Malaysia. 
        After detailed analysis, our project will be focusing on data from \"**Cases and Testing**\" and \"**Deaths**\".
        """)
    
    # Question 1
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
        'clusters.csv': clusters,
        'deaths_malaysia.csv': deaths_malaysia,
        'deaths_state.csv': deaths_state
    }
    selected_dataset = st.selectbox(
        "Dataset selection", [key for key in datasets])

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
    st.write("# 2. Correlation Analysis")
    st.markdown("Problem Statement: What are the states that exhibit strong correlation with Pahang and Johor?")

    st.markdown("To find the states that correlated to Johor and Pahang, we will use correlation heatmaps. We calculated the correlation between each states and show them in a heatmap. The higher the correlation score between that state and the target state (Johor or Pahang), the stronger the correlation between both states")

    st.markdown("It is shown that Kedah has a strong correlation with Pahang while Pulau Pinang and Sabah have high correlation with Johor")

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
    st.write("Problem Statement: What are the strong features to daily cases for Pahang, Kedah, Johor, and Selangor?")

    st.write('For this question, we have to determine the features that correlated to daily cases(`cases_new`) of Pahang, Kedah, Johor and Selangor. To get the results, we will be using correlation heatmaps and feature importance method to calculate the correlation score between each variables and target variable(`cases_new`)')

    st.write("## 3.1 Correlation Heatmap")

    st.write("A correlation heatmap is a graphical representation of the correlation matrix that represents the correlation between different variables.")
    st.markdown("""
    According to the heatmaps below, we could determine strong features to `cases_new` for Penang, Kedah, Johor and Selangor. If the value of the correlation score is higher or the color is brighter, that variable is stronger to `cases_new`.  

    For i) **Penang**, variable `cases_recovered` has the highest correlation which is 0.67.  
    For ii) **Kedah**, there are 3 variables that has high correlation to daily cases, which are `cases_recovered`, `rfk_ag` and `pcr` three of them get around 80 of correlation scores.  
    For iii) **Johor**, we found total 5 variables are highly correlated to daily cases. Five of them are `cases_recovered`, `rtk_ag`, `pcr`, `deaths_new` and `deaths_new_dod`. Five of them got 80% above of correlation scores.  
    Finally, for iv) **Selangor**, similar to Pahang, each of the variables did not show a very high correlation score to `cases_new`. The variable `deaths_new_dod` got the highest correlation score which is 0.62.
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
        im = Image.open('img/selangor_f.png')
        st.image(im, width=700, caption='Heatmap for Selangor')

    st.write('## 3.2 Feature Importance Method')
    st.write('Next, we will be using the feature importance method to determine the strong features of daily cases. Feature Importance will assign a score to each of the variables according to how they useful for predicting target variable. The higher the score a feature receives shows its importance to the variable.')
    st.markdown("""
    The algorithms used for our feature importance are:  
    1. Decision Tree Regression  
    2. Random Forest Regression

    For i) **Pahang**, it is shown that the feature importance for Decision Tree and Random Forest Regression is the almost same. The strongest feature of Pahang's daily cases is `cases_recovered`, followed by `pcr` and `rtk-ag`.  
    ii) **Kedah** has only one strong feature for its daily cases, which is `cases_recovered`.  
    Unlike Pahang and Kedah, iii) **Johor** has a different set of features affecting its daily cases under different regressors. Using a Decision Tree Regressor, it is shown that the top 3 features are `deaths_new`, `cases_recovered` and `rtk-ag`. On the other hand, while using a Random Forest Regressor, the top 3 features are `deaths_new_dod`, `cases_recovered`, `rtk-ag`.  
    Finally for iv) **Selangor**, the Decision Tree Regressor shows the top features for its daily cases are `cases_recovered`, `deaths_new_dod`. The results are also the same from Random Forest Regressor.
    """)

    stateQ3b = ["--- select a state ---", "Pahang", "Kedah", "Johor", "Selangor"]

    selected_stateQ3b = st.selectbox("Feature Importance", stateQ3b)

    if selected_stateQ3b == "Pahang":
        st.write("""
        *Decision Tree Regression* Feature Importance for Pahang  
        `cases_import` 0.0451  
        `cases_recovered` 0.5361  
        `deaths_bid` 0.0616  
        `deaths_bid_dod` 0.0297  
        `deaths_new` 0.0820  
        `deaths_new_dod` 0.0715  
        `pcr` 0.0522  
        `rtk-ag` 0.1217  
        """)
        im = Image.open('img/FI_Pahang_DTR.png')
        st.image(im, width=700, caption='')
        im = Image.open('img/FI_Pahang_DTR_mean.png')
        st.image(im, width=700, caption='')

        st.write("""
        Random Forest Regression Feature Importance for Pahang  
        `cases_import` 0.0172  
        `cases_recovered` 0.6276  
        `deaths_bid` 0.0155  
        `deaths_bid_dod` 0.0109  
        `deaths_new` 0.0347  
        `deaths_new_dod` 0.0483  
        `pcr` 0.1153  
        `rtk-ag` 0.1304  
        """)        
        im = Image.open('img/FI_Pahang_RFR.png')
        st.image(im, width=700, caption='')
        im = Image.open('img/FI_Pahang_RFR_mean.png')
        st.image(im, width=700, caption='')

    elif selected_stateQ3b == "Kedah":
        st.write("""
        Decision Tree Regression Feature Importance for Kedah  
        `cases_import` 0.0  
        `cases_recovered` 0.8312  
        `deaths_bid` 0.0025  
        `deaths_bid_dod` 0.0214  
        `deaths_new` 0.0163  
        `deaths_new_dod` 0.0650  
        `pcr` 0.0364  
        `rtk-ag` 0.0273  
        """)        
        im = Image.open('img/FI_Kedah_DTR.png')
        st.image(im, width=700, caption='')
        im = Image.open('img/FI_Kedah_DTR_mean.png')
        st.image(im, width=700, caption='')

        st.write("""
        Random Forest Regression Feature Importance for Kedah  
        `cases_import` 0.0001  
        `cases_recovered` 0.7551  
        `deaths_bid` 0.0010  
        `deaths_bid_dod` 0.0111  
        `deaths_new` 0.0305  
        `deaths_new_dod` 0.0520  
        `pcr` 0.0493  
        `rtk-ag` 0.1009  
        """)        
        im = Image.open('img/FI_Kedah_RFR.png')
        st.image(im, width=700, caption='')
        im = Image.open('img/FI_Kedah_RFR_mean.png')
        st.image(im, width=700, caption='')

    elif selected_stateQ3b == "Johor":
        st.write("""
        Decision Tree Regression Feature Importance for Johor  
        `cases_import` 0.0005  
        `cases_recovered` 0.1473  
        `deaths_bid` 0.0677  
        `deaths_bid_dod` 0.0328  
        `deaths_new` 0.0236  
        `deaths_new_dod` 0.6902  
        `pcr` 0.0228  
        `rtk-ag` 0.0150  
        """)        
        im = Image.open('img/FI_Johor_DTR.png')
        st.image(im, width=700, caption='')
        im = Image.open('img/FI_Johor_DTR_mean.png')
        st.image(im, width=700, caption='')

        st.write("""
        Random Forest Regression Feature Importance for Johor  
        `cases_import` 0.0026  
        `cases_recovered` 0.2431  
        `deaths_bid` 0.0263  
        `deaths_bid_dod` 0.0260  
        `deaths_new` 0.1027  
        `deaths_new_dod` 0.2743  
        `pcr` 0.0948  
        `rtk-ag` 0.2302  
        """)        
        im = Image.open('img/FI_Johor_RFR.png')
        st.image(im, width=700, caption='')
        im = Image.open('img/FI_Johor_RFR_mean.png')
        st.image(im, width=700, caption='')

    elif selected_stateQ3b == "Selangor":
        st.write("""
        Decision Tree Regression Feature Importance for Selangor  
        `cases_import` 0.0085  
        `cases_recovered` 0.4288  
        `deaths_bid` 0.0063  
        `deaths_bid_dod` 0.0480  
        `deaths_new` 0.0667  
        `deaths_new_dod` 0.4119  
        `pcr` 0.0164  
        `rtk-ag` 0.0134  
        """)        
        im = Image.open('img/FI_Selangor_DTR.png')
        st.image(im, width=700, caption='')
        im = Image.open('img/FI_Selangor_DTR_mean.png')
        st.image(im, width=700, caption='')

        st.write("""
        Random Forest Regression Feature Importance for Selangor  
        `cases_import` 0.0150  
        `cases_recovered` 0.2986  
        `deaths_bid` 0.0234  
        `deaths_bid_dod` 0.1567  
        `deaths_new` 0.0539  
        `deaths_new_dod` 0.3483  
        `pcr` 0.0624  
        `rtk-ag` 0.0418  
        """)        
        im = Image.open('img/FI_Selangor_RFR.png')
        st.image(im, width=700, caption='')
        im = Image.open('img/FI_Selangor_RFR_mean.png')
        st.image(im, width=700, caption='')


    # Question 4
    st.write("# 4. Regression and Classification")

    st.markdown("Problem Statement: Comparing regression and classification models to determine which model perform well in daily cases for Pahang, Kedah, Johor and Selangor")
    
    st.write("## 4.1 Regression")

    st.markdown("""
    Regression models that was used:  
    1. Linear Regression  
    2. Decision Tree Regressor  
    3. Random Forest Regressor  
    4. Support Vector Regressor  
    
    Evaluation matrics that was used:  
    1. R Square  
    2. Mean Absolute Error(MAE)  
    3. Root Mean Square Error(RMSE)  
    """)

    im = Image.open('img/regression.PNG')
    st.image(im, width=700, caption = 'Regression Model Evaluation')

    st.markdown('R Square values range from 0 to 1. The higher value of R Square indicates that the model fits the data better. On the other hand, both MAE and RMSE ranges from 0 to  ??? , and lower values are preferred. In our case, the R Square values of all 4 regressors are very close to 1. However, their MAE and RMSE are very high. We would consider **Decision Tree** and **Random Forest** as better regressors in this case as their R Square is higher and the errors are relatively lower than others.')
    

    st.write("## 4.2 Classification")

    st.markdown("""
    Classification models that was used:  
    1. K-Nearest Neighbors Classifier  
    2. Naive Bayes Classifier  
    3. Decision Tree Classifier  
    4. Random Forest Classifier  

    Performance evaluation metrics that was used:    
    1. Confusion Matrix  
    2. Precision, Recall, F1-score  
    3. ROC Curve  
    """)

    st.markdown('For classification, we categorize our target variale to three classes using binning, which are high, moderate and low daily new cases. ')

    st.markdown("""
    Applying SMOTE to balance the training dataset.

    SMOTE helps oversample minority classes in our data. In our case, low and moderate the minorities in cases_new_binned. We will apply SMOTE to our training data before fitting it into a model. This helps balance the class distribution during the training but not giving any additional information. SMOTE should only be applied to the training dataset (not the testing/validation set).

    If SMOTE is implemented prior to the train-test splitting, some of the synthetic data might end up in the testing/validation set, allowing the model to perform well at the moment. However, the model will underperform in production as it overfits to most of our synthetic data.

    References:
    [SMOTE for Imbalanced Classification with Python](https://machinelearningmastery.com/smote-oversampling-for-imbalanced-classification/#:~:text=This%20can%20be%20achieved%20by%20simply%20duplicating%20examples%20from%20the%20minority%20class%20in%20the%20training%20dataset%20prior%20to%20fitting%20a%20model.%20This%20can%20balance%20the%20class%20distribution%20but%20does%20not%20provide%20any%20additional%20information%20to%20the%20model.)

    """)

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

    st.markdown('For our classification models, the evaluation metrics we used are Confusion Matrix, Precision, Recall, F1-score and ROC curve. According to the evaluation metrics above, it is shown that our best model for classifying the daily cases for Pahang, Kedah, Johor and Selangor is the **Random Forest Classifier**, following up by **KNN** and Decision Tree. The Naive Bayes Classifier performs the poorest in this case.')

if __name__ == "__main__":
    main()
