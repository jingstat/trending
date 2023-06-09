import pandas as pd
from random import randint
from faker import Faker
from datetime import datetime, timedelta

# Instantiate a Faker instance for generating fake data
fake = Faker()

# Define lists to hold data
authors = []
article_ids = []
publish_dates = []
reads = []
days_since_publication = []
daily_reads = []
content_types = []
# Generate mock data
for _ in range(100):  # for 100 articles
    author = fake.name()
    article_id = fake.uuid4()  # using UUIDs for unique article IDs
    publish_date = fake.date_between(start_date='-1y', end_date='today')  # dates within the last year
    content = randint(0,3)
    # Calculate days since publication
    days_post_publication = (datetime.now().date() - publish_date).days

    # Generate daily reads
    for day in range(days_post_publication + 1):
        daily_read = randint(0, 100)  # random number of daily reads between 0 and 100
        authors.append(author)
        article_ids.append(article_id)
        publish_dates.append(publish_date)
        reads.append(daily_read)
        days_since_publication.append(day)
        daily_reads.append(daily_read)
        content_types.append(content)

# Create a DataFrame
df = pd.DataFrame({
    'Author': authors,
    'Article_ID': article_ids,
    'publish_date': publish_dates,
    'Day': days_since_publication,
    'Daily_Reads': daily_reads,
    'content_types':content_types
})

# Convert 'Publish_Date' to datetime
df['publish_date'] = pd.to_datetime(df['publish_date'])

# Print DataFrame
print(df)

df.to_csv('readership_data.csv')