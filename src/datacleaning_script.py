# %% [markdown]
# ## Import Libraries and Frameworks

# %%
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
import sklearn
import spacy
import pickle

from textblob import TextBlob
from spacytextblob.spacytextblob import SpacyTextBlob
import regex as re
from spacy.lang.en.stop_words import STOP_WORDS
from datetime import datetime

nlp = spacy.load("en_core_web_sm")
nlp.add_pipe('spacytextblob')

# %% [markdown]
# ## Pre-process functions

# %%
def get_mode(x):
    modes = pd.Series.mode(x)
    if len(modes) == 1:
        return modes[0]
    else:
        return "UNCLEAR"

# %%
def move_links_and_punc(df, X_col):
    """ 
    Create new columns 'word_count', 'num_links', 'has_links', and a clean version of the text column that has 
    urls, punctuation, and numbers removed
    
    Args
        df - pandas dataframe
        X_col - name of text column to be cleaned and extracted from
    """

    urlregex = r'(http\S+|www\S+)'
    numregex = r'\d+'
    puncregex = r'[^\w\s]'

    df['links'] = df[X_col].apply(lambda x: re.findall(urlregex, str(x)))
    df[X_col] = df[X_col].fillna({'data':''})
    df['clean_' + X_col] = df[X_col].replace(urlregex, '', regex=True).str.lower()
    df['clean_' + X_col] = df['clean_' + X_col].replace(puncregex, '', regex=True)
    df['clean_' + X_col] = df['clean_' + X_col].replace(numregex, '', regex=True)
    df['clean_' + X_col] = df['clean_' + X_col].apply(lambda x: "".join([" " if ord(i) < 32 or ord(i) > 126 else i for i in x]))
    df['clean_' + X_col] = df['clean_' + X_col].apply(lambda x: " ".join(x.strip() for x in x.split()))
    
    df['word_count'] = [len(x) for x in df['clean_' + X_col]]
    df['num_links'] = [len(x) for x in df['links']]
    df['has_links'] = [1 if x > 0 else 0 for x in df['num_links']]

# %%
def remove_stopwords(df, X_col):
    """ 
    Removes stopwords from the given text column in a dataframe
    
    Args
        df - pandas dataframe
        X_col - name of text column to be cleaned and extracted from
    """

    arr = []
    docs = nlp.pipe(df[X_col])
    for doc in docs:
        arr.append([str(tok.lemma_) for tok in doc if tok.text not in STOP_WORDS] )
    df[X_col] = arr

# %%
# Sentiment polarity and subjectivity functions
def sentiment_polarity(text):
    return text.apply(lambda Text: pd.Series(TextBlob(' '.join(Text)).sentiment.polarity))

def sentiment_subjectivity(text):
    return text.apply(lambda Text: pd.Series(TextBlob(' '.join(Text)).sentiment.subjectivity))

# %%


# %%
d = {'tweets':tweets}
with open('temp.pickle', 'wb') as handle:
    pickle.dump(d, handle, protocol=pickle.HIGHEST_PROTOCOL)

# %%
# Convert 'helpful' column into a trinary where each category is dependent on its average helpfulness rating
def fixHelpfulness(df, col, min=0.29, max=0.84):
    """
    Takes df, col
    0.84 and 0.29 are the cut-offs suggested by Twitter themselves
    """

    mask = df[col] >= max
    df.loc[mask, col] = 2
    mask = df[col].between(min, max, inclusive=False)
    df.loc[mask, col] = 1
    mask = df[col] <= min
    df.loc[mask, col] = 0

# %%
def drop_rows_with_empty_features(X, y):
    is_zero = [len(element) == 0 for element in X]
    print("dropping ", len(is_zero), " rows")
    X = [element for element, drop in zip(X, is_zero) if not drop]
    y = [element for element, drop in zip(y, is_zero) if not drop]
    return X, y

# %%
def get_mode(x):
    modes = pd.Series.mode(x)
    if len(modes) == 1:
        return modes[0]
    else:
        return "UNCLEAR"

# %%
def timestamp(x):
    try:
        dt = datetime.strptime(str(x), '%Y-%m-%d %H:%M:%S')
    except Exception as e:
        return
    epoch = datetime.utcfromtimestamp(0)
    return (dt - epoch).total_seconds() * 1000.0

# %%
notes = pd.read_csv('notes.tsv', sep="\t")
#ratings = pd.read_csv(ratings_filename, sep="\t")

# %%
notes.summary

# %%
ratings = pd.read_csv('ratings.tsv', sep="\t")


# %%
ratings

# %%
tweets = pd.read_csv('noted-tweets.csv', sep=",")

# %%


# %%
tweets.text.str.len().value_counts()

# %%
tweets[tweets.text.isna()]

# %%
def consolidate_files(notes_filename, ratings_filename, tweets_filename, start_date='01/01/2021', end_date='01/01/2022'):
    """
    Take in the data as provided by twitter and output
    a "ready for ML" version.

    Args
        notes_filename - string filename of the notes csv file
        ratings_filename - string filename of the ratings csv file 
        tweets_filename - string filename of the tweets csv file
        start_date - string date of the format month/day/year of the start bound of the timeframe
        end_date - string date of the format month/day/year of the end bound of the timeframe

    Output
        notesWithRatings - pandas dataframe joining notes and their corresponding ratings
        tweetsWithNotes - pandas dataframe joining tweets and their corresponding notes
    """
    
    
    # read files
    notes = pd.read_csv(notes_filename, sep="\t")
    ratings = pd.read_csv(ratings_filename, sep="\t")
    tweets = pd.read_csv(tweets_filename, sep=",")

    # adjust timeframes
    start_date = datetime.strptime(start_date, '%m/%d/%Y').timestamp() * 1000
    end_date = datetime.strptime(end_date, '%m/%d/%Y').timestamp() * 1000

    # print
    tweets.dropna(inplace=True, how='any')
    tweets['createdAtMillis'] = tweets['time'].apply(lambda x: timestamp(x[:19]))

    notes = notes[(notes.createdAtMillis.astype(int) > start_date) & (notes.createdAtMillis.astype(int) < end_date)]
    ratings = ratings[(ratings.createdAtMillis.astype(int) > start_date) & (ratings.createdAtMillis.astype(int) < end_date)]
    tweets = tweets[(tweets.createdAtMillis.astype(int) > start_date) & (tweets.createdAtMillis.astype(int) < end_date)]

    # consolidate dataframes
    ratingsWithNotes = notes.set_index('noteId').join(ratings.set_index('noteId'), lsuffix="_note", rsuffix="_rating", how='inner')
    average_ratings = ratings.groupby('noteId').mean()
    average_notes = notes[['tweetId', 'classification', 'believable', 'harmful', 'validationDifficulty']]
    tweet_id_to_mode = average_notes.groupby(['tweetId']).agg(get_mode)
    tweetsWithNotes = tweets.set_index('tweetId').join(tweet_id_to_mode, lsuffix="_tweet", rsuffix="_note", how='inner')
    notesWithRatings = notes.set_index('noteId').join(average_ratings, lsuffix="_note", rsuffix="_rating")

    tweetsWithNotes.dropna(inplace=True, how='any')
    notesWithRatings.dropna(inplace=True, how='any')

    # clean up text
    move_links_and_punc(notesWithRatings, 'summary')
    remove_stopwords(notesWithRatings, 'clean_summary')

    move_links_and_punc(tweetsWithNotes, 'text')
    remove_stopwords(tweetsWithNotes, 'clean_text')

    # create polarity and subjectivity columns
    notesWithRatings['polarity'] = sentiment_polarity(notesWithRatings['clean_summary'])
    notesWithRatings['subjectivity'] = sentiment_subjectivity(notesWithRatings['clean_summary'])

    tweetsWithNotes['polarity'] = sentiment_polarity(tweetsWithNotes['clean_text'])
    tweetsWithNotes['subjectivity'] = sentiment_subjectivity(tweetsWithNotes['clean_text'])

    # convert helpfulness score into Twitter specified divisinos
    fixHelpfulness(notesWithRatings, 'helpful')

    # create additional columns for nlp use
    notesWithRatings['clean_summary_as_str'] = notesWithRatings['clean_summary'].apply(lambda x: ' '.join(x))
    tweetsWithNotes['clean_text_as_str'] = tweetsWithNotes['clean_text'].apply(lambda x: ' '.join(x))

    return notesWithRatings, tweetsWithNotes

# %% [markdown]
# ### Pickle Dictionary Data for Use in Other Notebooks

# %%
dates = [
    '02/01/2021', '02/15/2021', '03/01/2021', '03/15/2021', '04/01/2021',
    '04/15/2021', '05/01/2021', '05/15/2021', 
    '06/01/2021', '06/15/2021',
    '07/01/2021', '07/15/2021',
    '08/01/2021', '08/15/2021',
    '09/01/2021', '09/15/2021',
    
]
dictionary = {}
for date in dates:
    print('processing ' + date)
    notesWithRatings, tweetsWithNotes = consolidate_files('notes-2.tsv', 'ratings-2.tsv', 'noted-tweets.csv', end_date=date)
    dictionary['notes ' + date] = notesWithRatings
    dictionary['tweets ' + date] = tweetsWithNotes

# %%
with open('processed-3.pickle', 'wb') as handle:
    pickle.dump(dictionary, handle, protocol=pickle.HIGHEST_PROTOCOL)

# %% [markdown]
# <a style='text-decoration:none;line-height:16px;display:flex;color:#5B5B62;padding:10px;justify-content:end;' href='https://deepnote.com?utm_source=created-in-deepnote-cell&projectId=c0e62a2c-7f45-414e-8164-5bf51e09d482' target="_blank">
# <img alt='Created in deepnote.com' style='display:inline;max-height:16px;margin:0px;margin-right:7.5px;' src='data:image/svg+xml;base64,PD94bWwgdmVyc2lvbj0iMS4wIiBlbmNvZGluZz0iVVRGLTgiPz4KPHN2ZyB3aWR0aD0iODBweCIgaGVpZ2h0PSI4MHB4IiB2aWV3Qm94PSIwIDAgODAgODAiIHZlcnNpb249IjEuMSIgeG1sbnM9Imh0dHA6Ly93d3cudzMub3JnLzIwMDAvc3ZnIiB4bWxuczp4bGluaz0iaHR0cDovL3d3dy53My5vcmcvMTk5OS94bGluayI+CiAgICA8IS0tIEdlbmVyYXRvcjogU2tldGNoIDU0LjEgKDc2NDkwKSAtIGh0dHBzOi8vc2tldGNoYXBwLmNvbSAtLT4KICAgIDx0aXRsZT5Hcm91cCAzPC90aXRsZT4KICAgIDxkZXNjPkNyZWF0ZWQgd2l0aCBTa2V0Y2guPC9kZXNjPgogICAgPGcgaWQ9IkxhbmRpbmciIHN0cm9rZT0ibm9uZSIgc3Ryb2tlLXdpZHRoPSIxIiBmaWxsPSJub25lIiBmaWxsLXJ1bGU9ImV2ZW5vZGQiPgogICAgICAgIDxnIGlkPSJBcnRib2FyZCIgdHJhbnNmb3JtPSJ0cmFuc2xhdGUoLTEyMzUuMDAwMDAwLCAtNzkuMDAwMDAwKSI+CiAgICAgICAgICAgIDxnIGlkPSJHcm91cC0zIiB0cmFuc2Zvcm09InRyYW5zbGF0ZSgxMjM1LjAwMDAwMCwgNzkuMDAwMDAwKSI+CiAgICAgICAgICAgICAgICA8cG9seWdvbiBpZD0iUGF0aC0yMCIgZmlsbD0iIzAyNjVCNCIgcG9pbnRzPSIyLjM3NjIzNzYyIDgwIDM4LjA0NzY2NjcgODAgNTcuODIxNzgyMiA3My44MDU3NTkyIDU3LjgyMTc4MjIgMzIuNzU5MjczOSAzOS4xNDAyMjc4IDMxLjY4MzE2ODMiPjwvcG9seWdvbj4KICAgICAgICAgICAgICAgIDxwYXRoIGQ9Ik0zNS4wMDc3MTgsODAgQzQyLjkwNjIwMDcsNzYuNDU0OTM1OCA0Ny41NjQ5MTY3LDcxLjU0MjI2NzEgNDguOTgzODY2LDY1LjI2MTk5MzkgQzUxLjExMjI4OTksNTUuODQxNTg0MiA0MS42NzcxNzk1LDQ5LjIxMjIyODQgMjUuNjIzOTg0Niw0OS4yMTIyMjg0IEMyNS40ODQ5Mjg5LDQ5LjEyNjg0NDggMjkuODI2MTI5Niw0My4yODM4MjQ4IDM4LjY0NzU4NjksMzEuNjgzMTY4MyBMNzIuODcxMjg3MSwzMi41NTQ0MjUgTDY1LjI4MDk3Myw2Ny42NzYzNDIxIEw1MS4xMTIyODk5LDc3LjM3NjE0NCBMMzUuMDA3NzE4LDgwIFoiIGlkPSJQYXRoLTIyIiBmaWxsPSIjMDAyODY4Ij48L3BhdGg+CiAgICAgICAgICAgICAgICA8cGF0aCBkPSJNMCwzNy43MzA0NDA1IEwyNy4xMTQ1MzcsMC4yNTcxMTE0MzYgQzYyLjM3MTUxMjMsLTEuOTkwNzE3MDEgODAsMTAuNTAwMzkyNyA4MCwzNy43MzA0NDA1IEM4MCw2NC45NjA0ODgyIDY0Ljc3NjUwMzgsNzkuMDUwMzQxNCAzNC4zMjk1MTEzLDgwIEM0Ny4wNTUzNDg5LDc3LjU2NzA4MDggNTMuNDE4MjY3Nyw3MC4zMTM2MTAzIDUzLjQxODI2NzcsNTguMjM5NTg4NSBDNTMuNDE4MjY3Nyw0MC4xMjg1NTU3IDM2LjMwMzk1NDQsMzcuNzMwNDQwNSAyNS4yMjc0MTcsMzcuNzMwNDQwNSBDMTcuODQzMDU4NiwzNy43MzA0NDA1IDkuNDMzOTE5NjYsMzcuNzMwNDQwNSAwLDM3LjczMDQ0MDUgWiIgaWQ9IlBhdGgtMTkiIGZpbGw9IiMzNzkzRUYiPjwvcGF0aD4KICAgICAgICAgICAgPC9nPgogICAgICAgIDwvZz4KICAgIDwvZz4KPC9zdmc+' > </img>
# Created in <span style='font-weight:600;margin-left:4px;'>Deepnote</span></a>


