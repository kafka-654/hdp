#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sat Jan 30 19:35:39 2021

@author: shashwat
"""

import streamlit as st
import pandas as pd
from sklearn.preprocessing import LabelEncoder
from sklearn.model_selection import train_test_split
from sklearn.svm import SVC
from sklearn.metrics import precision_score, recall_score
from sklearn.metrics import plot_confusion_matrix, plot_roc_curve, plot_precision_recall_curve
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
import numpy as np


def main():
    st.title("Heart Disease Classification Web App")
    st.sidebar.title("Heart Disease Classification Web App")
    st.markdown("Do you have a heart disease? ")
    st.sidebar.markdown("Do you have a Heart Disease?")
    
    def load_data():
        data = pd.read_csv("heart.csv")
        return data
    
    def split(df):
        y = df.target
        x = df.drop(columns = ["target"])
        x_train, x_test, y_train, y_test = train_test_split(x,y, test_size = 0.2, random_state = 42)
        return x_train, x_test, y_train, y_test
    
    def plot_metrics(metrics_list):
        if "Confusion Matrix" in metrics_list:
            st.subheader("Confusion Matrix")
            #fig, ax = plt.subplots()
            st.set_option('deprecation.showPyplotGlobalUse', False)
            plot_confusion_matrix(model, x_test, y_test, display_labels = class_names)
            st.pyplot()
        
        if "ROC curve" in metrics_list:
            st.subheader("ROC Curve")
            #fig, ax = plt.subplots()
            st.set_option('deprecation.showPyplotGlobalUse', False)
            plot_roc_curve(model, x_test, y_test)
            st.pyplot()
        
        if "Precision-Recall Curve" in metrics_list:
            st.subheader("Precion recall curve")
            st.set_option('deprecation.showPyplotGlobalUse', False)
            plot_precision_recall_curve(model, x_test, y_test)
            st.pyplot()
            

    df = load_data()
    x_train, x_test, y_train, y_test = split(df)
    class_names = ["Disease","No Disease"]
    
    if st.sidebar.checkbox("Show data set", False):
        st.subheader("Heart disease Data set")
        st.write(df)
        
    
            
            
            
            
            
            
if __name__ == '__main__':
    main()
        
        
        
        
