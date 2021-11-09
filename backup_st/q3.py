import streamlit as st
import numpy as np
import pandas as pd
from PIL import Image

st.header("Question iii\n")
st.markdown("#### What are the strong features to daily cases for Pahang, Kedah,Johor, and Selangor?\n")

st.markdown('For this questions we have to determine featuresa that correlated to daily cases(cases_new) of Pahang, Kedah, Johor and Selangor.\nTo get the results, we will using Correlation Heatmap and Feature Importance method to calculate the correlation score between each variables and target variables(cases_news)')

st.subheader("\nCorrelation Heatmap\n")

st.markdown("Correlation heatmap is a graphical representation of correlation matrix that representing correlation between different variables.\n")

im = Image.open('Pahang_f.png')
st.image(im,width=700, caption='Heatmap for Pahang')

im = Image.open('Kedah_f.png')
st.image(im,width=700, caption='Heatmap for Kedah')

im = Image.open('Johor_f.png')
st.image(im,width=700, caption='Heatmap for Johor')

im = Image.open('Selangor_f.png')
st.image(im,width=700, caption='Heatmap for Selangor')

st.markdown('According to the heatmaps above, we could determine strong features to cases_new for Penang, Kedah, Johor and Selangor.\n If the value of the correlation score is higher or color of that is brighter, that variable is stronger to cases_new.\nFor i) Penang, variable cases_recovered has highest correlation which is 0.67. For ii) Kedah, there are 3 variables has high correlation to daily cases, which are cases_recovered, rfk_ag and pcr three of them get around 80 of correlation scores.\n For iii) Johor, we found total 5 variables are highly correlated to daily cases. Five of them are cases_recovered, rtk_ag, pcr, deaths_new and deaths_new_dod. Five of them got 80% above of correlation scores.\nFinally, for iv) Selangor, similar like Pahang, each of the variables did not showns very high correlation score to cases_new. The variable deaths_new_dod got the highest correlation score which is 0.62.')

st.subheader('Feature Importance Method')
st.markdown('Next, we will using the Feature Importance Method to determine the strong features to daily cases. Feature Importance will assign a score\nto each of the variables according to how they useful for predicting target variable.\n If a feature get higher score, which mean it is stronger to daily cases')



