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
        \"[Open Data on COVID-19 in Malaysia](https://github.com/MoH-Malaysia/covid19-public)\" by Ministry of Health (MOH), Malaysia. 
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

    st.markdown("To find the states that correlated to Johor and Pahang, we will use correlation heatmap. If the correlation score between that state and `cases_new` of Johor or Pahang higher, the correlation between both states are stronger")

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

    st.write('For this questions we have to determine featuresa that correlated to daily cases(`cases_new`) of Pahang, Kedah, Johor and Selangor. To get the results, we will using Correlation Heatmap and Feature Importance method to calculate the correlation score between each variables and target variables(`cases_new`)')

    st.write("## 3.1 Correlation Heatmap")

    st.write("Correlation heatmap is a graphical representation of correlation matrix that representing correlation between different variables.")
    st.markdown("""
    According to the heatmaps above, we could determine strong features to cases_new for Penang, Kedah, Johor and Selangor. If the value of the correlation score is higher or color of that is brighter, that variable is stronger to `cases_new`.  

    For i) **Penang**, variable `cases_recovered` has highest correlation which is 0.67.  
    For ii) **Kedah**, there are 3 variables has high correlation to daily cases, which are `cases_recovered`, `rfk_ag` and `pcr` three of them get around 80 of correlation scores.  
    For iii) **Johor**, we found total 5 variables are highly correlated to daily cases. Five of them are `cases_recovered`, `rtk_ag`, `pcr`, `deaths_new` and `deaths_new_dod`. Five of them got 80% above of correlation scores.  
    Finally, for iv) **Selangor**, similar like Pahang, each of the variables did not showns very high correlation score to `cases_new`. The variable `deaths_new_dod` got the highest correlation score which is 0.62.
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

    st.write('## 3.2 Feature Importance Method')
    st.write('Next, we will using the Feature Importance Method to determine the strong features to daily cases. Feature Importance will assign a score to each of the variables according to how they useful for predicting target variable. If a feature get higher score, which mean it is stronger to daily cases.')
    st.markdown("""
    The algorithms used for our feature importance are:  
    1. Decision Tree Regression  
    2. Random Forest Regression

    For i) **Pahang**, it is shown that the feature importance for Decision Tree and Random Forest Regression are the almost same. The strongest feature to Pahang's daily cases is `cases_recovered`, followed by `pcr` and `rtk-ag`.  
    ii) **Kedah** has only one strong feature for its daily cases, which is `cases_recovered`.  
    Unlike Pahang and Kedah, iii) **Johor** has a different set of features affecting its daily cases under different regressors. Using a Decision Tree Regressor, it is shown that the top 3 features are `deaths_new`, `cases_recovered` and `rtk-ag`. On the other hand, while using a Random Forest Regressor, the top 3 features are `deaths_new_dod`, `cases_recovered`, `rtk-ag`.  
    Finally for iv) **Selangor**, the Decision Tree Regressor shows the top features for its daily cases are `cases_recovered`, `deaths_new_dod`. The results are also the same from Random Forest Regressor.
    """)

    stateQ3b = ["--- select a state ---", "Pahang", "Kedah", "Johor", "Selangor"]

    selected_stateQ3b = st.selectbox("Feature Importance", stateQ3b)

    if selected_stateQ3b == "Pahang":
        st.write("""
        *Decision Tree Regression* Feature Importance for Pahang\n
        cases_import 0.04511602026722506\n
        cases_recovered 0.5360548216702808\n
        deaths_bid 0.06162983722921824\n
        deaths_bid_dod 0.02973137344102241\n
        deaths_new 0.08202430683380657\n
        deaths_new_dod 0.07151403069456355\n
        pcr 0.05221738887632135\n
        rtk-ag 0.12171222098756189\n
        """)
        im = Image.open('img/FI_Pahang_DTR.png')
        st.image(im, width=700, caption='')
        im = Image.open('img/FI_Pahang_DTR_mean.png')
        st.image(im, width=700, caption='')

        st.write("""
        Random Forest Regression Feature Importance for Pahang\n
        cases_import 0.01723921010897433\n
        cases_recovered 0.6276390087273518\n
        deaths_bid 0.015491471589392309\n
        deaths_bid_dod 0.010938141951370915\n
        deaths_new 0.03467304155907633\n
        deaths_new_dod 0.04830340135894632\n
        pcr 0.11529903225639324\n
        rtk-ag 0.13041669244849488\n
        """)        
        im = Image.open('img/FI_Pahang_RFR.png')
        st.image(im, width=700, caption='')
        im = Image.open('img/FI_Pahang_RFR_mean.png')
        st.image(im, width=700, caption='')

    elif selected_stateQ3b == "Kedah":
        st.write("""
        Decision Tree Regression Feature Importance for Kedah\n
        cases_import 0.0\n
        cases_recovered 0.8311860951915561\n
        deaths_bid 0.002459136208146091\n
        deaths_bid_dod 0.021386564796206917\n
        deaths_new 0.016309891129043633\n
        deaths_new_dod 0.06496135863151967\n
        pcr 0.03636748049051394\n
        rtk-ag 0.02732947355301361\n
        """)        
        im = Image.open('img/FI_Kedah_DTR.png')
        st.image(im, width=700, caption='')
        im = Image.open('img/FI_Kedah_DTR_mean.png')
        st.image(im, width=700, caption='')

        st.write("""
        Random Forest Regression Feature Importance for Kedah\n
        cases_import 7.081645824801608e-05\n
        cases_recovered 0.7551162305892843\n
        deaths_bid 0.0010360153993133603\n
        deaths_bid_dod 0.011088899076453308\n
        deaths_new 0.030544648480122102\n
        deaths_new_dod 0.05196528770787843\n
        pcr 0.04930299217059195\n
        rtk-ag 0.10087511011810854\n
        """)        
        im = Image.open('img/FI_Kedah_RFR.png')
        st.image(im, width=700, caption='')
        im = Image.open('img/FI_Kedah_RFR_mean.png')
        st.image(im, width=700, caption='')

    elif selected_stateQ3b == "Johor":
        st.write("""
        Decision Tree Regression Feature Importance for Johor\n
        cases_import 0.0005497956300280646\n
        cases_recovered 0.14731250440997815\n
        deaths_bid 0.06772418384277851\n
        deaths_bid_dod 0.03279485958535056\n
        deaths_new 0.023604470301506166\n
        deaths_new_dod 0.6901827555148919\n
        pcr 0.022834914516777865\n
        rtk-ag 0.014996516198688813\n
        """)        
        im = Image.open('img/FI_Johor_DTR.png')
        st.image(im, width=700, caption='')
        im = Image.open('img/FI_Johor_DTR_mean.png')
        st.image(im, width=700, caption='')

        st.write("""
        Random Forest Regression Feature Importance for Johor\n
        cases_import 0.0026314469176732425\n
        cases_recovered 0.24305102322241612\n
        deaths_bid 0.026321666662441533\n
        deaths_bid_dod 0.025996740667788076\n
        deaths_new 0.10272266630182862\n
        deaths_new_dod 0.2742556347774977\n
        pcr 0.0948067529687426\n
        rtk-ag 0.2302140684816121\n
        """)        
        im = Image.open('img/FI_Johor_RFR.png')
        st.image(im, width=700, caption='')
        im = Image.open('img/FI_Johor_RFR_mean.png')
        st.image(im, width=700, caption='')

    elif selected_stateQ3b == "Selangor":
        st.write("""
        Decision Tree Regression Feature Importance for Selangor\n
        cases_import 0.008468007809660056\n
        cases_recovered 0.42881779977251544\n
        deaths_bid 0.006346634965632681\n
        deaths_bid_dod 0.048048194907221224\n
        deaths_new 0.06665286277682352\n
        deaths_new_dod 0.41192766697951727\n
        pcr 0.01635994548068081\n
        rtk-ag 0.013378887307949033\n
        """)        
        im = Image.open('img/FI_Selangor_DTR.png')
        st.image(im, width=700, caption='')
        im = Image.open('img/FI_Selangor_DTR_mean.png')
        st.image(im, width=700, caption='')

        st.write("""
        Random Forest Regression Feature Importance for Selangor\n
        cases_import 0.015008221072619217\n
        cases_recovered 0.2985571519905454\n
        deaths_bid 0.02338748881150938\n
        deaths_bid_dod 0.15667795778602508\n
        deaths_new 0.05389411924171667\n
        deaths_new_dod 0.34830486533896327\n
        pcr 0.06235003696542771\n
        rtk-ag 0.041820158793193396\n
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

    im = Image.open('img/regression.png')
    st.image(im, width=700, caption = 'Regression Model Evaluation')

    st.markdown('R Square values ranges from 0 to 1. The higher value of R Square indicates that the model fits the data better. On the other hand, both MAE and RMSE ranges from 0 to  ∞ , and lower values are preferred. In our case, the R Square values of all 4 regressor are very close to 1. However, their MAE and RMSE are very high. We would consider **Decision Tree** and **Random Forest** as better regressors in this case as their R Square is higher and the errors are relatively lower than others.')
    

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
    Applying SMOTE to balance the training dataset.\n

    SMOTE helps oversample minority classes in our data. In our case, low and moderate the minorities in cases_new_binned. We will apply SMOTE to our training data before fitting into a model. This helps balance the class distribution during the training but not giving any additional information. SMOTE should only be applied to the training dataset (not testing/validation set).\n

    If SMOTE is implemented prior to the train-test splitting, some of the synthetic data might end up in the testing/validation set, allowing the model to perform well at the moment. However, the model will underperform in production as it overfits to most of our synthetic data.\n

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

    st.markdown('For classification model, the evaluation matrix we used are Confusion Matrix, precision, recall, F1-score and ROC curve. According to the evaluation matrix above, it is shown that our best model for classifying the daily cases for Pahang, Kedah, Johor and Selangor is the **Random Forest Classifier**, following up by **KNN** and Decision Tree. The Naive Bayes Classifier performs the poorest in this case.')

if __name__ == "__main__":
    main()
