# Author: R James
# Date:   2020-03-26T17:19:24-07:00
# Last modified by:   R James
# Last modified time: 2020-04-10T11:08:44-07:00
# Description: cleaning r/incels data for topic models
# Notes: Conducted using tutorial from: https://towardsdatascience.com/end-to-end-topic-modeling-in-python-latent-dirichlet-allocation-lda-35ce4ed6b3e0

# Loading necessary packages

import pandas as pd
import re

# Reading in incels posts and comments

df1 = pd.read_csv('C:/Users/asus/Google Drive/Projects/Red Pill/Data/Working Data/RP_Incels_posts_2017_01.csv')
df2 = pd.read_csv('C:/Users/asus/Google Drive/Projects/Red Pill/Data/Working Data//RP_Incels_comments_2017_03.csv')

### 'Linking' posts and comments

# Dropping 'retrieved on' and 'subreddit' columns because they're unneeded

df1 = df1.drop(columns=['subreddit','retrieved_on','created_utc'])
df2 = df2.drop(columns=['subreddit', 'created_utc'])

# Adding column to reflect whether text was post or comment

df1['post_type'] = 'post'
df2['post_type'] = 'comment'

# Making remaining column names and order consistent across both data frames

df2['num_comments'] = '0' # Adding 'num_comments' to comments (this will be 0)

df2 = df2.reindex(columns=['author','score','num_comments','link_id','body','post_type']) # Matching column order for dataframes

df1.columns = ['author','score','num_comments','id','body','post_type'] # Matching comment column names to post column names
df2.columns = ['author','score','num_comments','id','body','post_type']

# Truncating link_id in comments to match post id

df2['id'].astype(str).str[3:]

# Combining data

df = pd.concat([df1, df2])

# Deleting rows with removed, deleted, or blank comments

df = df.dropna(axis=0, how='any', thresh=None) # Missing data
df = df[df.body != '[deleted]'] # Posts with deleted text
df = df[df.body != '[removed]'] # Posts with removed text

# Number of unique posters

df['author'].nunique()

# Exporting to new .csv

df.to_csv('RP_incels_2017_03.csv', index=False)
