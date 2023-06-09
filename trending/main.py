import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn import metrics
import matplotlib.pyplot as plt
import streamlit as st

# Load your dataset here
df = pd.read_csv('readership_data.csv')

# Get unique article IDs
article_ids = df['Article_ID'].unique()

# Sidebar for user input
st.sidebar.title('Article Readership')
selected_article_id = st.sidebar.selectbox('Select an article', article_ids)

# Filter dataframe by selected article ID
filtered_df = df[df['Article_ID'] == selected_article_id]

# Display the readership for the article
st.write(f'Readership for Article ID {selected_article_id}')
st.line_chart(filtered_df['Daily_Reads'])

# Train a simple linear regression model
X = filtered_df['Day'].values.reshape(-1,1)
y = filtered_df['Daily_Reads'].values.reshape(-1,1)

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=0)
regressor = LinearRegression()
regressor.fit(X_train, y_train)

# To make predictions on the test data
y_pred = regressor.predict(X_test)

df = pd.DataFrame({'Actual': y_test.flatten(), 'Predicted': y_pred.flatten()})
st.write(f'Predicted Readership for Article ID {selected_article_id}')
st.line_chart(df)

# Plotting the regression line
plt.scatter(X_test, y_test,  color='gray')
plt.plot(X_test, y_pred, color='red', linewidth=2)
plt.show()
