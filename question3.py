import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

def main():
    st.write("""
    # COVID-19 in Malaysia
    """)

    # cases and testing
    cases_malaysia = pd.read_csv('dataset/cases_and_testing/cases_malaysia.csv')
    cases_state = pd.read_csv('dataset/cases_and_testing/cases_state.csv')
    clusters = pd.read_csv('dataset/cases_and_testing/clusters.csv')
    tests_malaysia = pd.read_csv('dataset/cases_and_testing/tests_malaysia.csv')
    tests_state = pd.read_csv('dataset/cases_and_testing/tests_state.csv')


    # Sidebar
    st.sidebar.header("Select a Dataset")
    datasets = {
        '--- select a dataset ---': 0,
        'cases_malaysia.csv': cases_malaysia,
        'cases_state.csv': cases_state,
        'tests_malaysia.csv': tests_malaysia,
        'tests_state.csv': tests_state,
        'clusters.csv': clusters
    }
    selected_dataset = st.sidebar.selectbox("Cases and Testing", [key for key in datasets])

    df = datasets.get(selected_dataset)
    if not isinstance(df, pd.DataFrame):
        st.write("Please select a dataset from the sidebar to continue.") 
        return None
    
    st.write("""
        ## Dataset
    """)
    st.write(f"Selected Dataset: `{selected_dataset}`")
    st.write(df)
    
    st.write("""
        ## Missing value
    """)
    missing_value = df[df.isna().values.any(axis=1)]
    rows = missing_value.shape[0]
    if rows > 0:
        st.write(f"There are {rows} rows with missing values")
        st.write("Identified missing value in each column:")
        st.write(df.isna().sum())
    else:
        st.write("There are no missing values in this dataset")


if __name__ == "__main__":
    main()