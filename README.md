# Sentiments towards U.S. Airlines: A February 2015 Analysis

This GitHub repository contains code and analysis for a sentiment analysis project on U.S. airlines. The project analyzes Twitter data from February 2015 to understand customer sentiments toward various airlines. This README provides an overview of the project, its structure, and how to use the code.

## Table of Contents
- [Import Libraries](#import-libraries)
- [Data Loading: Loading Data from "Tweets.csv"](#data-loading-loading-data-from-tweetscsv)
- [Checking Dataset Dimensions](#checking-dataset-dimensions)
- [Data Preview: First 5 Rows of the Loaded Data](#data-preview-first-5-rows-of-the-loaded-data)
- [Dataset Information](#dataset-information)
- [Handling Missing Values: Context and Decisions](#handling-missing-values-context-and-decisions)
    - [Handling Missing Values in `user_timezone`](#user_timezone)
- [Top 5 Most Common User Timezones and Tweet Locations](#top-5-most-common-user-timezones-and-tweet-locations)
- [Analyzing Sentiments in Tweets](#analyzing-sentiments-in-tweets)
- [Visualizing Sentiments in Tweets](#visualizing-sentiments-in-tweets)
- [Plotting the Distribution of Text Length for Positive Sentiment Tweets](#plotting-the-distribution-of-text-length-for-positive-sentiment-tweets)
- [Analyzing Text Length in Positive Sentiment Tweets](#analyzing-text-length-in-positive-sentiment-tweets)
- [Analyzing Text Length in Negative Sentiment Tweets](#analyzing-text-length-in-negative-sentiment-tweets)
- [Airline Sentiments for Each Airline](#airline-sentiments-for-each-airline)
- [Visualizing Negative Sentiment Reasons](#visualizing-negative-sentiment-reasons)
- [Breakdown of Negative Sentiment Reasons by Airline](#breakdown-of-negative-sentiment-reasons-by-airline)
- [Comparative Analysis of Negative Sentiment Reasons by Airline](#comparative-analysis-of-negative-sentiment-reasons-by-airline)
- [Percentage of Negative Sentiment Customers by Airline](#percentage-of-negative-sentiment-customers-by-airline)
- [Most Used Words in Positive and Negative Tweets](#most-used-words-in-positive-and-negative-tweets)
- [Wordcloud for Negative Sentiments of Tweets](#wordcloud-for-negative-sentiments-of-tweets)
- [Wordcloud for Positive Sentiments of Tweets](#wordcloud-for-positive-sentiments-of-tweets)
- [Reasons for Negative Sentimental Tweets for Each Airline](#what-are-the-reasons-for-negative-sentimental-tweets-for-each-airline)
- [Text Preprocessing](#text-preprocessing)
    - [Descriptive Statistics of Tweet Length](#descriptive-statistics-of-tweet-length)
    - [Exploring Tweet Text](#exploring-tweet-text)
    - [Text Preprocessing Functions](#text-preprocessing-functions)
- [Sentiment Analysis Using VADER SentimentIntensityAnalyzer](#sentiment-analysis-using-vader-sentimentintensityanalyzer)
    - [Sentiment Analysis Results Using VADER SentimentIntensityAnalyzer](#sentiment-analysis-results-using-vader-sentimentintensityanalyzer)
    - [Preprocessing for Word Clouds by Sentiment](#preprocessing-for-word-clouds-by-sentiment)
- [Machine Learning Approach](#machine-learning-approach)
- [Model Comparison](#model-comparison)
- [Word2Vec](#word2vec)
    - [Creating Word2Vec Average Vectors](#creating-word2vec-average-vectors)
    - [Word2Vec-Based Sentiment Analysis](#word2vec-based-sentiment-analysis)
    - [Text Tokenization and Padding](#text-tokenization-and-padding)
    - [Data Splitting](#data-splitting)
    - [Model Training](#model-training)
    - [Model Evaluation on Test Set](#model-evaluation-on-test-set)
    - [Confusion Matrix](#confusion-matrix)
- [Sentiment Analysis with Random Forest Classifier](#sentiment-analysis-with-random-forest-classifier)
    - [Data Preparation](#data-preparation)
    - [Applying TFIDF on Cleaned Tweets](#now-apply-tfidf-on-cleaned-tweets)
    - [Handling Class Imbalance with SMOTE](#handling-class-imbalance-with-smote)
    - [Splitting Data into Train & Test](#split-data-into-train--test)
    - [Training a Random Forest Classifier](#training-a-random-forest-classifier)
    - [Evaluating the Random Forest Classifier](#evaluating-the-random-forest-classifier)
- [Training the XGBoost Classifier](#training-the-xgboost-classifier)
- [Evaluating the XGBoost Classifier](#evaluating-the-xgboost-classifier)
- [Training the Gradient Boosting Classifier](#training-the-gradient-boosting-classifier)
- [Accuracy of the Gradient Boosting Classifier Model](#accuracy-of-the-gradient-boosting-classifier-model)
- [Training a Support Vector Machine (SVM) Classifier](#training-a-support-vector-machine-svm-classifier)
- [Evaluating Support Vector Machine (SVM) Classifier](#evaluating-support-vector-machine-svm-classifier)
- [Training a Multinomial Naive Bayes Classifier](#training-a-multinomial-naive-bayes-classifier)
- [Evaluating Multinomial Naive Bayes Classifier](#evaluating-multinomial-naive-bayes-classifier)
- [Training a Decision Tree Classifier](#training-a-decision-tree-classifier)
- [Evaluating the Decision Tree Classifier](#evaluating-the-decision-tree-classifier)
- [Random Forest Classifier Evaluation](#random-forest-classifier-evaluation)

## Import Libraries
This section imports the necessary Python libraries and modules required for the analysis.

## Data Loading: Loading Data from "Tweets.csv"
Here, we load the dataset from "Tweets.csv" for analysis.

## Checking Dataset Dimensions
We check the dimensions (rows and columns) of the loaded dataset.

## Data Preview: First 5 Rows of the Loaded Data
This section displays the first 5 rows of the dataset to provide an initial glimpse of the data.

## Dataset Information
Provides information about the dataset, including data types, null values, and summary statistics.

## Handling Missing Values: Context and Decisions
Explains the context of missing values in the dataset and the decisions made to handle them.

### Handling Missing Values in `user_timezone`
Explains the specific handling of missing values in the `user_timezone` column.

## Top 5 Most Common User Timezones and Tweet Locations
Analyzes and displays the top 5 most common user timezones and tweet locations.

## Analyzing Sentiments in Tweets
Explains the sentiment analysis approach used in the project.

## Visualizing Sentiments in Tweets
Visualizes the sentiments in tweets using various charts and graphs.

## Plotting the Distribution of Text Length for Positive Sentiment Tweets
Analyzes the text length distribution for positive sentiment tweets.

## Analyzing Text Length in Positive Sentiment Tweets
Explores and analyzes text length in positive sentiment tweets.

## Analyzing Text Length in Negative Sentiment Tweets


Explores and analyzes text length in negative sentiment tweets.

## Airline Sentiments for Each Airline
Analyzes the sentiments for each airline.

## Visualizing Negative Sentiment Reasons
Visualizes the reasons behind negative sentiments.

## Breakdown of Negative Sentiment Reasons by Airline
Breaks down negative sentiment reasons by airline.

## Comparative Analysis of Negative Sentiment Reasons by Airline
Compares negative sentiment reasons among different airlines.

## Percentage of Negative Sentiment Customers by Airline
Calculates the percentage of negative sentiment customers for each airline.

## Most Used Words in Positive and Negative Tweets
Analyzes the most used words in positive and negative tweets.

## Wordcloud for Negative Sentiments of Tweets
Creates word clouds for negative sentiments in tweets.

## Wordcloud for Positive Sentiments of Tweets
Creates word clouds for positive sentiments in tweets.

## Reasons for Negative Sentimental Tweets for Each Airline
Analyzes the reasons for negative sentimental tweets for each airline.

## Text Preprocessing
Explains the text preprocessing steps undertaken in the project.

### Descriptive Statistics of Tweet Length
Provides descriptive statistics of tweet length after preprocessing.

### Exploring Tweet Text
Explores the preprocessed tweet text.

### Text Preprocessing Functions
Details the functions used for text preprocessing.

## Sentiment Analysis Using VADER SentimentIntensityAnalyzer
Explains the use of the VADER SentimentIntensityAnalyzer for sentiment analysis.

### Sentiment Analysis Results Using VADER SentimentIntensityAnalyzer
Displays sentiment analysis results using VADER.

### Preprocessing for Word Clouds by Sentiment
Prepares data for generating word clouds by sentiment.

## Machine Learning Approach
Describes the machine learning approach for sentiment analysis.

## Model Comparison
Compares different machine learning models.

## Word2Vec
Explains the use of Word2Vec for sentiment analysis.

### Creating Word2Vec Average Vectors
Details the process of creating Word2Vec average vectors.

### Word2Vec-Based Sentiment Analysis
Performs sentiment analysis using Word2Vec.

### Text Tokenization and Padding
Explains text tokenization and padding.

### Data Splitting
Describes how data is split into training and testing sets.

### Model Training
Details the training of the Word2Vec-based sentiment analysis model.

### Model Evaluation on Test Set
Evaluates the performance of the Word2Vec-based model on the test set.

### Confusion Matrix
Displays the confusion matrix for the Word2Vec-based model.

## Sentiment Analysis with Random Forest Classifier
Explains sentiment analysis using a Random Forest Classifier.

### Data Preparation
Describes data preparation steps for the Random Forest model.

### Applying TFIDF on Cleaned Tweets
Applies TFIDF on cleaned tweets.

### Handling Class Imbalance with SMOTE
Explains how class imbalance is handled using SMOTE.

### Splitting Data into Train & Test
Splits the data into training and testing sets.

### Training a Random Forest Classifier
Details the training of the Random Forest Classifier.

### Evaluating the Random Forest Classifier
Evaluates the performance of the Random Forest Classifier.

## Training the XGBoost Classifier
Details the training of the XGBoost Classifier.

## Evaluating the XGBoost Classifier
Evaluates the performance of the XGBoost Classifier.

## Training the Gradient Boosting Classifier
Details the training of the Gradient Boosting Classifier.

## Accuracy of the Gradient Boosting Classifier Model
Calculates the accuracy of the Gradient Boosting Classifier model.

## Training a Support Vector Machine (SVM) Classifier
Details the training of the Support Vector Machine (SVM) Classifier.

## Evaluating Support Vector Machine (SVM) Classifier
Evaluates the performance of the Support Vector Machine (SVM) Classifier.

## Training a Multinomial Naive Bayes Classifier
Details the training of the Multinomial Naive Bayes Classifier.

## Evaluating Multinomial Naive Bayes Classifier
Evaluates the performance of the Multinomial Naive Bayes Classifier.

## Training a Decision Tree Classifier
Details the training of the Decision Tree Classifier.

## Evaluating the Decision Tree Classifier
Evaluates the performance of the Decision Tree Classifier.

## Random Forest Classifier Evaluation
Provides an evaluation of the Random Forest Classifier model.

Please refer to the corresponding sections for more details on each aspect of the project. Feel free to explore the code and analysis in this repository to gain insights into customer sentiments towards U.S. airlines in February 2015.
