import streamlit as st
import numpy as np
import pandas as pd
from PIL import Image

st.header("Question iv")
st.markdown("#### Comparing classification and regression model to determine which model perform well in daily cases for Pahang, Kedah, Johor and Selangor")

st.markdown('For classification, models we will use are K-Nearest Neighbors Classifier, Naive Bayes Clasifier,Decision Tree Classifier and Random Forest Classifier. We divide our target variale to three classes according to binning, which are high, moderate and low daily new cases. For regression, models we will use are Linear Regression, Decision Tree Regressor, Random Forest Regressor and Support Vector Regressor')

st.subheader("Classification")

im = Image.open('confusion1.png')  
st.image(im,width=1000, caption='')

im = Image.open('confusion2.png')  
st.image(im,width=1000, caption='Confusion Matrix')

im = Image.open('classification1.png')  
st.image(im,width=1000, caption='')

im = Image.open('classification2.png')  
st.image(im,width=1000, caption='Evalaution for classification models')

im = Image.open('ROC.png')
st.image(im, width=800, caption = 'ROC curve')



st.markdown('For classification model, the evaluation matrix we used are Confusion Matrix, precision, recall, F1-score and ROC curve. According to the evaluation matrix above, it is shown that our best model for classifying the daily cases for Pahang, Kedah, Johor and Selangor is the Random Forest Classifier, following up by KNN and Decision Tree. The Naive Bayes Classifier performs the poorest in this case.')

st.subheader("Regression")

im = Image.open('regression.png')
st.image(im, width=1200, caption = 'Regression Model Evaluation')

st.markdown('R Square values ranges from 0 to 1. The higher value of R Square indicates that the model fits the data better. On the other hand, both MAE and RMSE ranges from 0 to  ∞ , and lower values are preferred.In our case, the R Square values of all 4 regressor are very close to 1. However, their MAE and RMSE are very high. We would consider Decision Tree and Random Forest as better regressors in this case as their R Square is higher and the errors are relatively lower than others.')

