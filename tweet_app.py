#!/usr/bin/env python
# coding: utf-8
import streamlit as st
import pandas as pd
import numpy as np
import plotly.express as px
from plotly.subplots import make_subplots
import plotly.graph_objects as go
from wordcloud import WordCloud, STOPWORDS
import matplotlib.pyplot as plt
import nltk
from nltk.corpus import stopwords
import os
import warnings
warnings.filterwarnings("ignore")

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

col1, mid= st.beta_columns([1,20])
with col1:
    st.image('twitter_logo.svg',width=100)

DATA_URL = ("Tweets.csv")
def session():
    session+=1
    return session
def button():
    command="streamlit run ./live_app.py"
    os.system(command)
   
            
if st.sidebar.button("Home"):
    button()

st.title("Sentiment Analysis")
st.markdown("Review of US Airlines")
st.sidebar.title("Exploratory Data analysis(Visualization)")


@st.cache(persist=True)
def load_data():
    
    data = pd.read_csv(DATA_URL)
    data['tweet_created'] = pd.to_datetime(data['tweet_created'])
    return data

data = load_data()


st.sidebar.subheader("Display random tweet")
random_tweet = st.sidebar.radio('Sentiment', ('positive', 'neutral', 'negative'))
st.sidebar.markdown(data.query("airline_sentiment == @random_tweet")[["text"]].sample(n=1).iat[0, 0])

st.sidebar.markdown("### Total Number of tweets by sentiment")
select = st.sidebar.selectbox('Type of visualization', ['Bar plot', 'Pie chart'], key='1')
sentiment_count = data['airline_sentiment'].value_counts()
sentiment_count = pd.DataFrame({'Sentiment':sentiment_count.index, 'Tweets':sentiment_count.values})





#move to plotting
if not st.sidebar.checkbox("Hide", True): #by defualt hide the checkbar
    st.markdown("### Number of tweets by sentiment")
    if select == 'Bar plot':
        fig = px.bar(sentiment_count, x='Sentiment', y='Tweets', color='Tweets', height=500)
        st.plotly_chart(fig)
    else:
        fig = px.pie(sentiment_count, values='Tweets', names='Sentiment')
        st.plotly_chart(fig)



#Interactive bar plots
st.sidebar.subheader("Total number of tweets for each airline")
each_airline = st.sidebar.selectbox('Type of visualization', ['Bar plot', 'Pie chart'], key='2')
airline_sentiment_count = data.groupby('airline')['airline_sentiment'].count().sort_values(ascending=False)
airline_sentiment_count = pd.DataFrame({'Airline':airline_sentiment_count.index, 'Tweets':airline_sentiment_count.values.flatten()})
if not st.sidebar.checkbox("Close", True, key='2'):
    if each_airline == 'Bar plot':
        st.subheader("Total number of tweets for each airline")
        fig_1 = px.bar(airline_sentiment_count, x='Airline', y='Tweets', color='Tweets', height=500)
        st.plotly_chart(fig_1)
    if each_airline == 'Pie chart':
        st.subheader("Total number of tweets for each Airline")
        fig_2 = px.pie(airline_sentiment_count, values='Tweets', names='Airline')
        st.plotly_chart(fig_2)



#No. of Tweets by sentiment of Each Airline
@st.cache(persist=True)
def plot_sentiment(airline):
    df = data[data['airline']==airline]
    count = df['airline_sentiment'].value_counts()
    count = pd.DataFrame({'Sentiment':count.index, 'Tweets':count.values.flatten()})
    return count



st.sidebar.subheader("Breakdown airline by sentiment")
choice = st.sidebar.multiselect('choose airlines', ('US Airways','United','American','Southwest','Delta','Virgin America'))
if len(choice) > 0:
    st.subheader("Breakdown airline by sentiment")
    breakdown_type = st.sidebar.selectbox('Type of visualization', ['Pie chart', 'Bar plot', ], key='3')
    fig_3 = make_subplots(rows=1, cols=len(choice), subplot_titles=choice)
    if breakdown_type == 'Bar plot':
        for i in range(1):
            for j in range(len(choice)):
                fig_3.add_trace(
                    go.Bar(x=plot_sentiment(choice[j]).Sentiment, y=plot_sentiment(choice[j]).Tweets, showlegend=False),
                    row=i+1, col=j+1
                )
        fig_3.update_layout(height=600, width=800)
        st.plotly_chart(fig_3)
    else:
        fig_3 = make_subplots(rows=1, cols=len(choice), specs=[[{'type':'domain'}]*len(choice)], subplot_titles=choice)
        for i in range(1):
            for j in range(len(choice)):
                fig_3.add_trace(
                    go.Pie(labels=plot_sentiment(choice[j]).Sentiment, values=plot_sentiment(choice[j]).Tweets, showlegend=True),
                    i+1, j+1
                )
        fig_3.update_layout(height=600, width=800)
        st.plotly_chart(fig_3)
        
st.sidebar.subheader("Breakdown airline by sentiment")
choice = st.sidebar.multiselect('Pick airlines', ('US Airways','United','American','Southwest','Delta','Virgin America'), key=0)
if len(choice) > 0:
    choice_data = data[data.airline.isin(choice)]
    fig_0 = px.histogram(
                        choice_data, x='airline', y='airline_sentiment',
                         histfunc='count', color='airline_sentiment',
                         facet_col='airline_sentiment', labels={'airline_sentiment':'tweets'},
                          height=600, width=800)
    st.plotly_chart(fig_0)





from nltk.corpus import stopwords
#Text Preprocessing and Word Cloud visualization of sentiments
#stop_words.update(',',';','!','?','.','(',')','$','#','+',':','...',' ','')
words=data['text'].dropna().apply(nltk.word_tokenize)
from PIL import Image
st.set_option('deprecation.showPyplotGlobalUse', False)
t_mask = np.array(Image.open("twitter_mask.png"))
st.sidebar.header("Word Cloud")
word_sentiment = st.sidebar.radio('Display word cloud for what sentiment?', ('positive', 'neutral', 'negative'))
if not st.sidebar.checkbox("Close", True, key='3'):
    st.subheader('Word cloud for %s sentiment' % (word_sentiment))
    df = data[data['airline_sentiment']==word_sentiment]
    words = ' '.join(df['text'])
    processed_words = ' '.join([word for word in words.split() if 'http' not in word and not word.startswith('@') and word != 'RT'])
    wordcloud = WordCloud(stopwords=STOPWORDS, background_color='white',contour_width=3,mask=t_mask,contour_color='steelblue').generate(processed_words)
    plt.imshow(wordcloud)
    plt.xticks([])
    plt.yticks([])
    st.pyplot();

    







































