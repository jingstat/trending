import pandas as pd
from random import randint
from faker import Faker
from datetime import datetime, timedelta
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn import metrics
import matplotlib.pyplot as plt
import streamlit as st
import plotly.express as px
from datetime import datetime

# Load your dataset here
#df = pd.read_csv('readership_data.csv')
# Print DataFrame
#print(df)

st.title('Article Readership Dashboard')

# load data
@st.cache  # this decorator helps cache the data to speed up the app
def load_data():
    df = pd.read_csv('readership_data.csv')
    df['publish_date'] = pd.to_datetime(df['publish_date'])
    df['publish_date'] = df['publish_date'].dt.date
    return df

df = load_data()
if st.checkbox('Show raw data'):
    st.subheader('Raw data')
    st.write(df)

# Filter by date range
start_date = st.sidebar.date_input('Start date', df['publish_date'].min())
print(start_date)
end_date = st.sidebar.date_input('End date', df['publish_date'].max())

df = df[(df['publish_date'] >= start_date) & (df['publish_date'] <= end_date)]
df = df.sort_values(['Article_ID', 'Day'],ascending=True)

# Filter by content type
content_types = st.sidebar.multiselect(
    'content_types',
    options=df['content_types'].unique().tolist(),
    default=df['content_types'].unique().tolist()
)
df = df[df['content_types'].isin(content_types)]
df['cumulative_reads'] = df.groupby('Article_ID')['Daily_Reads'].cumsum()

if not df.empty:
    color_attribute = st.selectbox("Select attribute for line color", ('Article_ID', 'content_types'))
    fig = px.line(df, x='Day', y='cumulative_reads', color=color_attribute, line_group='Article_ID', hover_name='Article_ID')
    fig.update_layout(title='Daily reads over time per document',
                          xaxis_title='Day',
                          yaxis_title='Cumulative Reads')
    st.plotly_chart(fig)

# Get unique article IDs
article_ids = df['Article_ID'].unique()
# Sidebar for user input
st.sidebar.title('Article Readership')
selected_article_id = st.sidebar.selectbox('Select an article', article_ids)
# Filter dataframe by selected article ID
df1 = df[df['Article_ID'] == selected_article_id]

# Daily reads over time
st.subheader('Daily reads over time')
fig = px.line(df1, x='Day', y='Daily_Reads', color='content_types', title='Daily reads over time')
st.plotly_chart(fig)

# Articles by author
st.subheader('Articles by author')
fig = px.bar(df['Author'].value_counts().reset_index(), x='Author', y='count', title='Articles by author')
st.plotly_chart(fig)
