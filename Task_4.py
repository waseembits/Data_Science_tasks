import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from wordcloud import WordCloud

# Set display and style
pd.set_option('display.max_columns', None)
sns.set(style="whitegrid")

df = pd.read_csv("movies.csv")

# Quick look at dataset
print(df.shape)
print(df.info())
df.head()

df.drop_duplicates(inplace=True)

# Handle missing values
df.fillna({
    'Genre': 'Unknown',
    'Lead Studio': 'Unknown',
    'Audience score %': df['Audience score %'].mean(),
    'Profitability': df['Profitability'].mean(),
    'Rotten Tomatoes %': df['Rotten Tomatoes %'].mean(),
    'Worldwide Gross': '$0'
}, inplace=True)

# Clean Worldwide Gross column (remove $ and convert to float)
df['Worldwide Gross'] = df['Worldwide Gross'].replace('[\$,]', '', regex=True).astype(float)

# Rename columns for convenience
df.rename(columns={
    'Film': 'title',
    'Genre': 'genre',
    'Lead Studio': 'studio',
    'Audience score %': 'audience_score',
    'Rotten Tomatoes %': 'rotten_tomatoes',
    'Year': 'release_year'
}, inplace=True)

# Final check
print(df.head())
print(df.info())

total_movies = df.shape[0]
print(f"Total number of movies: {total_movies}")

# Most common genres
common_genres = df['genre'].value_counts().head(10)
print("\nMost common genres:\n", common_genres)

# Top 10 highest-rated movies (based on Audience Score)
top_rated = df.sort_values(by='audience_score', ascending=False).head(10)
print("\nTop 10 highest-rated movies:\n", top_rated[['title', 'audience_score', 'genre']])

# Average profitability
avg_profitability = df['Profitability'].mean()
print(f"\nAverage Profitability: {avg_profitability:.2f}")

# Average Rotten Tomatoes score
avg_rotten = df['rotten_tomatoes'].mean()
print(f"Average Rotten Tomatoes Score: {avg_rotten:.2f}%")

# Average Worldwide Gross
avg_gross = df['Worldwide Gross'].mean()
print(f"Average Worldwide Gross: ${avg_gross:.2f} million")

