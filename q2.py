import streamlit as st
import numpy as np
import pandas as pd
from PIL import Image

st.header("Question ii\n")
st.markdown("#### What are the states that exhibit strong correlation with Pahang and Johor?\n")

st.markdown("To find the states that correlated to Johor and Pahang, we will use correlation heatmap.\n If the correlation score between that state and cases_new of Johor or Pahang higher, the correlation between both states are stronger")

im = Image.open('Pahang_state.png')
st.image(im,width=1000, caption='Heatmap for Pahang with other states')

im = Image.open('johor_state.png')
st.image(im,width=1000, caption='Heatmap for Johor with other states')

