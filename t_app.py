#!/usr/bin/env python
# coding: utf-8

# In[1]:
import string
import pickle
import tweepy
import streamlit as st
import pandas as pd
import numpy as np
import plotly.express as px
from plotly.subplots import make_subplots
import plotly.graph_objects as go
from wordcloud import WordCloud, STOPWORDS
import matplotlib.pyplot as plt
import nltk
import re
import os
from nltk.corpus import stopwords
from textblob import Word
english_stopwords = stopwords.words("english")
#stml.balloons()
from cleaner import *
from sklearn.feature_extraction.text import TfidfVectorizer,CountVectorizer
import time 
from textblob import *
# st.markdown(
#     """
#     <style>
#       .main{
#       background-color:#E1E8ED;
#       }
#       </style>

#     """,
#     unsafe_allow_html=True
# ) 
st.markdown(
    """
<style>
.sidebar.sidebar-content {
}
</style>
""",
    unsafe_allow_html=True,
)

# col1, mid= st.beta_columns([25,20])
# with col1:
st.image('twitter.png',use_column_width=True)

# #stml.balloons()
st.title("Real Time Tweet Analyser")
st.markdown(" Department of Computer Science")

def button():
    command="streamlit run ./live_app.py"
    os.system(command)
    os.system(exit(1))

if st.sidebar.button("Home"):
    button()
    
st.sidebar.title(" Real Time Sentiment Analysis of Tweet")




    
    
st.sidebar.markdown("Sentiment Analysis Dashboard")

# In[ ]:


@st.cache(allow_output_mutation=True)
def get_tweets(user_input):
    api=tweeter_auth() 
    posts = api.user_timeline(screen_name=user_input, count =200, lang ="en", since_id=2021-1-12)
#     cursor=tweepy.Cursor(api.user_timeline,id=user_input,tweet_mode="extended").items(100000000)
    return posts;

def tweeter_auth():
    try:
        consumer_key = "g0RcxBR6U7VxCICaDtTXtjZZy"
        consumer_secret = "gIW3LUXOZc3YNy6zDhu7EOudI0QPYzVZBKXxDFUGvk8MasscZj"
        access_key= "1393065638956994561-wC7ME2iMImQlTKhK4AOO5vLAaLcvtV"
        access_secret = "RwqSHVO3gDtjIwE8gmonG3dkBK5izSDFVSsiQKBzZiweY"
    except KeyError:
        sys.stderr.write("Twitter_*environment variable not set\n")
        sys.exit(1)
    auth = tweepy.OAuthHandler(consumer_key, consumer_secret)
    auth.set_access_token(access_key, access_secret)
    api=tweepy.API(auth,wait_on_rate_limit = True)
    return api



user_input=st.text_input("Input Twitter Screen_Name")
tweets=get_tweets(user_input)
df = pd.DataFrame([tweet.text for tweet in tweets], columns=['Tweets'])
df.to_csv("livetweet.csv")
df=pd.read_csv("livetweet.csv")
#st.write(df.head(10))
df['Processed_Tweet'] = df['Tweets'].apply(lambda x: preprocess(x))

df['Tokenized_Tweet'] = df['Processed_Tweet'].apply(lambda x: tokenize(x))

nan_value = float("NaN")

df.replace("", nan_value, inplace=True)

#drop the row that contain null value

df.dropna(inplace=True)




def getSubjectivity(text):
    return TextBlob(text).sentiment.subjectivity

# Create a function to get the polarity
def getPolarity(text):
    return  TextBlob(text).sentiment.polarity


# Create two new columns 'Subjectivity' & 'Polarity'
df['Subjectivity'] = df['Tweets'].apply(getSubjectivity)
df['Polarity'] = df['Tweets'].apply(getPolarity)

def getAnalysis(score):
      if score < 0:
        return 'Negative'
      elif score == 0:
        return 'Neutral'
      else:
        return 'Positive'
    
df['Sentiment'] = df['Polarity'].apply(getAnalysis)

df['label']=df['Sentiment'].replace({'Negative':0,'Neutral':1,'Positive':2})










# tokens=df['Processed_Tweet']
# # cv=CountVectorizer( max_features=800)
# # CV=cv.fit_transform(tokens)
# from sklearn.feature_extraction.text import TfidfVectorizer

# tfidf=TfidfVectorizer(ngram_range=(1,2),norm='l2',max_df=0.5, min_df=1,max_features=800,stop_words='english')

# tfidf_matrix=tfidf.fit_transform(tokens)

# CV= pd.DataFrame(tfidf_matrix.todense())




# # df.replace("", nan_value, inplace=True)

# # #drop the row that contain null value

# # df.dropna(inplace=True)

# filename = 'lr_model.sav'
# lr_model = pickle.load(open(filename, 'rb'))
   

# df['label']=lr_model.predict(CV)

# df['Sentiment']=df['label'].replace({0:'Negative', 1:'Neutral',2:'Positive'})

# # st.write(df.head(200))
# # st.write(df.shape)
# st.write(CV.shape)
with st.spinner('Analyzing.......'):
        time.sleep(1)
        st.success('Done!') 

    ########################################*****************VISUALIZATION*************########################################################

st.sidebar.subheader("Display random tweet of " + user_input.capitalize())
random_tweet = st.sidebar.radio('Sentiment', ('Positive', 'Neutral', 'Negative'))
st.sidebar.markdown(df.query("Sentiment == @random_tweet")[["Tweets"]].sample(n=1).iat[0, 0])


from PIL import Image

st.sidebar.markdown("Total Number of tweets by sentiment")
select = st.sidebar.selectbox('Type of visualization', ['Bar plot', 'Pie chart'], key='2')
sentiment_count = df['label'].value_counts()
sentiment_count = pd.DataFrame({'Sentiment':sentiment_count.index, 'Tweets':sentiment_count.values})
#move to plotting
if not st.sidebar.checkbox("Hide", True): #by defualt hide the checkbar
    st.markdown("Number of tweets by sentiment")
    if select == 'Bar plot':
        fig = px.bar(sentiment_count, x='Sentiment', y='Tweets', color=('Positive', 'Neutral', 'Negative'), height=500)
        st.plotly_chart(fig)
    else:
        fig = px.pie(sentiment_count, values='Tweets', names=('Positive', 'Neutral', 'Negative'))
        st.plotly_chart(fig)
        
        
        
                      #**********************WORDCLOUD***********************************
            
#, width=800, height=640
t_mask = np.array(Image.open("twitter_mask.png"))
st.set_option('deprecation.showPyplotGlobalUse', False)       
st.sidebar.header("Visualize Tweets by Word Cloud")
word_sentiment = st.sidebar.radio('Display Word Cloud for what sentiment?', ('Positive', 'Neutral', 'Negative'))
if not st.sidebar.checkbox("Close", True, key='3'):
    st.subheader('Word cloud for %s sentiment of ' % (word_sentiment))
    dfw = df[df['Sentiment']==word_sentiment]
    words = ' '.join(df['Processed_Tweet'])
    processed_words = ' '.join([word for word in words.split() if 'http' not in word and not word.startswith('@') and word != 'RT'])
    wordcloud = WordCloud(stopwords=STOPWORDS, background_color='white',contour_width=3,mask=t_mask,contour_color='steelblue').generate(processed_words)
    plt.imshow(wordcloud)
    plt.xticks([])
    plt.yticks([])
    st.pyplot();







