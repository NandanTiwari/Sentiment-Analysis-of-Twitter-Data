#!/usr/bin/env python
# coding: utf-8

# In[1]:

import streamlit as st
import sys
import pandas as pd
import os


# In[2]:
st.set_option('deprecation.showPyplotGlobalUse', False)
st.enableXsrfProtection = True

import tweepy


st.markdown(
    """
    <style>
      .main{
      background-color:#E1E8ED;
      }
      </style>

    """,
    unsafe_allow_html=True
) 

st.header('WELCOME TO MY PROJECT SENTIMENTAL ANALYSIS')
st.write("-----------BY NANDAN TIWARI--------------")
import streamlit as st
from datetime import date

today = date.today()
today=today.strftime("%d/%m/%Y")
st.write("DATE:-",today)

st.image("twitter.png",use_column_width=True)




# Add a selectbox to the sidebar:
add_selectbox = st.sidebar.selectbox('Select Project Running Environment',["Home",'Real time sentimental analysis',"Analysis sentiment Training data"])



def load_homepage() -> None:
    for i in range(2):
        st.write(" ")
    st.header("🎲 Dashbord for Sentiment Analysis of Twitter Data")
    st.write("Sentiment Analysis is the process of extracting the opinion or emotion from text data.   "
             "So Sometimes its called opinion mining or emotion AI. "
             "Moreover, it felt like a nice opportunity to see how much information can be "
             "extracted from relatively simple data.")
   
   
    
if add_selectbox=='Home':
    load_homepage()    

   
    

elif (add_selectbox=='Real time sentimental analysis'):
    import os
    command="streamlit run ./t_app.py"
    os.system(command)
    
elif (add_selectbox=='Analysis sentiment Training data'):
    import os
    command="streamlit run ./tweet_app.py"
    os.system(command)




# In[6]:


