import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.neighbors import KNeighborsClassifier
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.cluster import KMeans
import itertools

#Import datasets for manipulation
movies = pd.read_csv("C:\\Users\Brian\Desktop\EECS_731\Project3\Data\movies.csv")
ratings = pd.read_csv("C:\\Users\Brian\Desktop\EECS_731\Project3\Data\\ratings.csv")
tags = pd.read_csv("C:\\Users\Brian\Desktop\EECS_731\Project3\Data\\tags.csv")

#Merge the datasets into a single dataset
temp1 = pd.merge(movies, ratings, on=["movieId"], how="inner")
print(temp1.head())
temp2 = pd.merge(temp1, tags, on=["movieId"], how="inner")
print(temp2.head())

#Delete columns that won't be used
del temp2["timestamp_y"]
del temp2["timestamp_x"]
del temp2["userId_x"]
del temp2["userId_y"]
#Print the current dataset
print(temp2.head())

#Total the number of keywords that are found in the dataset
def numWords(data, col, lists):
    keyword_count = dict()
    for i in lists: keyword_count[i] = 0
    for lists_keywords in data[col].str.split('|'):
        if type(lists_keywords) == float and pd.isnull(lists_keywords): continue
        for j in lists_keywords: 
            if pd.notnull(j): keyword_count[j] += 1
    keyword_occurences = []
    for k,v in keyword_count.items():
        keyword_occurences.append([k,v])
    keyword_occurences.sort(key = lambda x:x[1], reverse = True)
    return keyword_occurences, keyword_count

#Set the labels and where the string splits
labels = set()
for s in temp2['genres'].str.split('|').values:
    labels = labels.union(set(s))

#Call to numWords, and displaying the number of times each word is in the dataset
numberOfWords = numWords(temp2, 'genres', labels)
print(numberOfWords)

#Plot the top 5 keywords
temp2.boxplot(column='rating', figsize=(10,5), return_type='both')
totalTags = temp2['tag'].value_counts()
print("\n", totalTags.head(10))

#Color scheme for the above graph
colors = ["#ffb612", "#e31837", "#0051ba", "#e8000d", "#512888", "#c0c0c0", "#e377c2", "#7f7f7f", "#bcbd22", "#17becf"]
totalTags[:5].plot(kind='bar', figsize=(5,2),color=colors)

#Get the mean of the ratings for each movie and display the first 10
average_ratings= temp2.groupby('movieId', as_index=False).mean()
del average_ratings['movieId']
print("\n", average_ratings.head(10))

#Display the number of movies per rating
movies_by_rating = temp2[['movieId','rating']].groupby('movieId').count()
print("\n", movies_by_rating.head())

#Setup the engine for the KNeighborsClassifier
engine = KNeighborsClassifier(n_neighbors=20)

#Data points for training
data_points = average_ratings[['rating']].values

#labels for training
labels = average_ratings.index.values

#Display the data and labels being used
print("\nData points(average ratings for the movies):\n", data_points)
print("\nLabels:\n ", labels)

#Fit the data and make predictions based on KNeighbors classification
engine.fit(data_points, labels)
pred = engine.predict(average_ratings)
print("\nPredictions based on KNeighbors Classifier: ", pred)

#Setup our models for KMeans clustering
features = temp2[['movieId','rating']]
def doKmeans(X, n_cluster = 8):
    model = KMeans(n_cluster)
    model.fit(X)
    cluster_labels = model.predict(X)
    center = model.cluster_centers_
    return (cluster_labels, center)

cluster_labels,center = doKmeans(features, 2)
kmeans = pd.DataFrame(cluster_labels)
features.insert((features.shape[1]),'kmeans', kmeans)

#Display the KMeans data in the form of a scatter graph
fig = plt.figure()
graph = fig.add_subplot(111)

colors = itertools.cycle(["r", "b", "g"])

scatter = graph.scatter(temp2['movieId'],features['rating'],
                     c=kmeans[0])
graph.set_title('K-Means Clustering')
graph.set_xlabel('Movie ID')
graph.set_ylabel('User Rating')
   
plt.colorbar(scatter)