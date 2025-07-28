# Created by R James on 5.1.2020
# Last updated: N/A
# Description: Reducing RP_incels_topicsentences_01.csv to only include topic of interests, and then creating random samples of texts for coding.
# Notes: Topic of interest is #27, which appears to capture themes related to race

import pandas as pd

df = pd.read_csv('C:/Users/asus/Google Drive/Projects/Red Pill/Data/SOC560 Analysis/Results/RP_incels_topicsentences_01.csv')

# Dropping rows on irrelevant topics

df = df[df.Dominant_Topic == 27.0]

# Dropping additional columns

df = df.drop(labels=None, columns=['Dominant_Topic', 'Perc_Contribution', 'Topic_Keywords'])

# There are too many texts to read, so I want to take a random sample. However, because I want to be able to code more than one random sample if necessary without overlap, I will create 20 random samples, each containing 791 cases.

random_samples = ["01", "02", "03", "04", "05", "06", "07", "08", "09", "10", "11", "12", "13", "14", "15", "16", "17" , "18", "19", "20"]

for i in random_samples:
    split = df.sample(n=791)
    split.to_csv('Results/Text Samples/RP_incels_randomsample_'+i+'.txt')
    df = df.drop(split.index)
