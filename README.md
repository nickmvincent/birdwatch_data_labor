# Introduction

Birdwatch is Twitter's pilot program to employ the use of volunteers to label tweets and notes along a spectrum of categories including misleading, harmful, difficulty of verification etc. Currently the program is in its alpha stage, only open to verified users located in the United States and selected for activity and political diversity but with the information gathered throughout the pilot Twitter has said that it will make executive decisions about its potential global rollout. 

Twitter has made available data gathered from this program into ready-for-ML .csv files. Birdwatch data is separated into two types: notes and note ratings, with daily updated tables available for download for both types.

# User Flow of Birdwatch and Data Labor Perspective

There are two aspects of data labor from the user flow created by Twitter. The first aspect is that Birdwatchers are able to create a 'note' on any public tweet and give a useful summary as to what the user has found to be misleading or worthy of a note. The second aspect is that Birdwatchers can then access notes left on public tweets by other Birdwatchers and leave a 'rating', a purely categorical labeling of that note. All of this value is generated from the effort of unwaged laborers who are a part of the program.

# How to use the Notebook

This notebook contains insights into the data and separates them into sections based on different information gathered. When looking to run this notebook from scratch using an updated dataset, here are the steps. 

1. Download daily updated data from https://twitter.github.io/birdwatch/contributing/download-data/ 
2. Run `twitterAPI.ipynb`, which scrapes Twitter for newly noted tweets and addsinformation into noted-tweets.csv
3. Run `datacleaning` notebook which joins data into two types of dataframes: tweets and corresponding notes, and notes and corresponding ratings. This notebook also cleans up text data through vectorization, the removal of links, punctuation, and numbers, and separates joined data into predictors and predictions.
4. Run `predictions` for ML experiments
5. Run `visualizations` to visualize label frequencies and other descriptive stats
6. Run `viz_ml` to reproduce figures from the paper (requires `predictions` to be run first)

Other files in this repo:
* `experiment_helpers.py` has important helper code for experiments. 
* `additional_explorations` is a scratchpad for exploring the data, includes an example of exploratory interface for viewing complex topic model results. 
* `descriptive` notebook has more descriptive stats
* `spacy_test` is just a notebook for making sure spacy is installed properly. 