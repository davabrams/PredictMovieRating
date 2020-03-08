# -*- coding: utf-8 -*-
"""
Created on Sun Sep 15 13:42:14 2019

@author: Dav Abrams
"""

import numpy as np
import matplotlib.pyplot as plt 
import matplotlib.image as mpimg
import datetime as dt
import tensorflow as tf
import pandas as pd
from tensorflow.keras import initializers
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Input, Dense, Dropout, Activation
from tensorflow.keras.utils import plot_model
from tensorflow.keras.optimizers import SGD, RMSprop
from sklearn.metrics import confusion_matrix
from sklearn import preprocessing
from sklearn.preprocessing import StandardScaler

import itertools

# fix random seed for reproducibility
seed = 7
np.random.seed(seed)

# %% bring data from disk to memory
title_basics = pd.read_csv('C:\\Users\\Dav\\Downloads\\title_basics.tsv', sep='\t')
#title_episode = pd.read_csv('C:\\Users\\Dav\\Downloads\\title_episode.tsv', sep='\t') #this contains TV data, irrelevant to this task
title_ratings = pd.read_csv('C:\\Users\\Dav\\Downloads\\title_ratings.tsv', sep='\t') #this data set provides the rating of the movie, with primary key tconst
title_akas = pd.read_csv('C:\\Users\\Dav\\Downloads\\title_akas.tsv', sep='\t')  #this data set provides the movie titles using primary key titleId

#title_crew = pd.read_csv('C:\\Users\\Dav\\Downloads\\title_crew.tsv', sep='\t')
#title_principals = pd.read_csv('C:\\Users\\Dav\\Downloads\\title_principals.tsv', sep='\t')
#name_basics = pd.read_csv('C:\\Users\\Dav\\Downloads\\name_basics.tsv', sep='\t')

# %%  inprocess the data - inner join the tables and remove some data that may confuse the regressor

mvp_dataset = title_basics[title_basics['titleType'] == 'movie'] #select only movies

#figure out the titles with the fewest ratings and throw them out (item #9)
fewest_rated_titles = title_ratings.sort_values(by=['numVotes'])['numVotes'].tolist()[round(len(title_ratings)*(10/100))]

title_ratings_removeLowVotes = title_ratings[title_ratings['numVotes'] > fewest_rated_titles]

mvp_dataset = pd.merge(mvp_dataset, title_ratings_removeLowVotes, on='tconst', how='inner') #add info on ratings BUT ONLY if the movie has more than 10 ratings
mvp_dataset = pd.merge(mvp_dataset, title_akas, left_on='tconst', right_on='titleId',how='inner') #add info on title name

#remove some fields we don't want right now
mvp_dataset = mvp_dataset.drop(columns=["ordering", "primaryTitle", "region", "originalTitle", "titleType", "titleId", "endYear", "attributes", "language"])

#so now these are our fields:
#    'tconst'                  -  unique key (leave it, but its not an input to the model)
#    'title'                   -  string, doesnt go into model

#    'isAdult'                 -  boolean, ready for model
#    'isOriginalTitle'         -  boolean, ready for model
#    'startYear'               -  int, ready for model
#    'runtimeMinutes'          -  int, ready for model
#    'numVotes'                -  int, ready for model

#    'averageRating'           -  float, ready for model (as output)

#    'types'                   -  string, needs to be encoded
#    'genres'                  -* string, comma separated, needs to be encoded


#just check that this is in here 
BestMovieEver = mvp_dataset[mvp_dataset['title'] == 'Hackers']  # (it is)

# %% cool, now we need an encoder to turn fields such as "types" and "genres" into categories

# **this is absolutely one of the worst ways to do this!!!**
# there is definitely an easy way to combine the two loops into one simple loop, or use some package!
# but it works.  just takes some time. (11a)
genreCategories=[]
for i_row in range(0,len(mvp_dataset['genres'])):                     #iterate through each row
    list_of_genres_in_row = mvp_dataset['genres'][i_row].split(",")   #parse the comma separated values
    for j_row in range(0,len(list_of_genres_in_row)):
        if list_of_genres_in_row[j_row] not in genreCategories:
            genreCategories.append(list_of_genres_in_row[j_row])

genreEncoded=np.zeros((len(mvp_dataset), len(genreCategories)))
for i_row in range(0,len(mvp_dataset['genres'])):                     #iterate through each row
    list_of_genres_in_row = mvp_dataset['genres'][i_row].split(",")   #parse the comma separated values
    for j_row in range(0,len(list_of_genres_in_row)):
        genreEncoded[i_row][genreCategories.index(list_of_genres_in_row[j_row])] = 1;


typeCategories=[]
for i_row in range(0,len(mvp_dataset['types'])):                     #iterate through each row
    list_of_types_in_row = mvp_dataset['types'][i_row].split("\x02")   #parse the comma separated values
    for j_row in range(0,len(list_of_types_in_row)):
        if list_of_types_in_row[j_row] not in typeCategories:
            typeCategories.append(list_of_types_in_row[j_row])

typeEncoded=np.zeros((len(mvp_dataset), len(typeCategories)))
for i_row in range(0,len(mvp_dataset['types'])):                     #iterate through each row
    list_of_types_in_row = mvp_dataset['types'][i_row].split("\x02")   #parse the comma separated values
    for j_row in range(0,len(list_of_types_in_row)):
        typeEncoded[i_row][typeCategories.index(list_of_types_in_row[j_row])] = 1;

# %%Set up a multiple linear regressor model by selecting inputs and outputs
y = mvp_dataset['averageRating']


x0 = np.array(mvp_dataset['isAdult'], dtype='uint8')
x1 = np.array(mvp_dataset['isOriginalTitle'])
failTitle = np.where(x1 == '\\N')[0]
for i in range(len(failTitle)):
    x1[failTitle[i]] = 0
x1 = np.array(x1, dtype='uint8')

x2 = np.array(mvp_dataset['startYear'])
#there are mixed values of int and string and '\\N' which need to be fixed (11b)
failYear = [];
for i in range(len(x2)):
    try:
        x2[i] = int(x2[i])
    except:
        x2[i] = 0
        failYear.append(i)
yMean = x2.mean()
for i in range(len(failYear)): #for the missing data, give it the average
    x2[failYear[i]] = yMean
x2 = np.array(x2, dtype='uint8')

x3 = np.array(mvp_dataset['runtimeMinutes'])
failTime = [];
for i in range(len(x3)):
    try:
        x3[i] = int(x3[i])
    except:
        x3[i] = 0
        failTime.append(i)
tMean = round(np.mean(x3))
for i in range(len(failTime)): #for the missing data, give it the average
    x3[failTime[i]] = tMean
x3 = np.array(x3, dtype='uint8')
    
x4 = np.array(mvp_dataset['numVotes'], dtype='uint8')
x5 = genreEncoded
x6 = typeEncoded

# stack the inputs and outputs (11c)

X = np.stack([x0, x1, x2, x3, x4])
for i_col in range(0, x5.shape[1]-1): #we don't add the last col as it would over-define the system
    X = np.vstack((X, x5[:, i_col]))
for i_col in range(0, x6.shape[1]-1): #we don't add the last col as it would over-define the system
    X = np.vstack((X, x6[:, i_col]))
X = X.T
#Y = the output metric (averageRating)
Y = np.stack(y).T

Z = np.array(mvp_dataset['title'])


# %% standardize / normalize our inputs.  this will be useful for our mixed data types. (11d)
sc_X = StandardScaler()
X = sc_X.fit_transform(X)


#Y = preprocessing.scale(Y)
Y = (Y - 5.5)/4.5; #histogram shows this data is somewhat normal distribution at about 6.25
#plt.plot(np.histogram(Y, bins=25)[0])
# %% split test and train sets (20% test size works for me) (11e)
from sklearn.model_selection import train_test_split
X_train, X_test, Y_train, Y_test = train_test_split(X, Y, test_size = 0.2, random_state = 0)

# %% USING KERAS 

# create model
#model.add(Dropout(0.5, input_shape=(X.shape[1],)))
model = Sequential()
model.add(Dense(25, input_dim=X.shape[1],kernel_initializer='random_normal'))
model.add(Dense(15, activation='relu', kernel_initializer='random_normal'))
model.add(Dense(5, activation='relu', kernel_initializer='random_normal'))
model.add(Dense(1, activation='linear',))

# Compile model
optim = RMSprop(lr=0.0001, rho=0.9, epsilon=None, decay=0.0)
model.compile(optimizer=optim, loss='mse', metrics=['mse', 'mae']) 
# Fit the model
history = model.fit(X_train, Y_train, validation_data=(X_test, Y_test), epochs=150, batch_size=10000, verbose=1)
# evaluate the model
scores = model.evaluate(X_test, Y_test)

#correct for score whitening
score_MSE = scores[1] * 4.5
score_MAE = scores[2] * 4.5
print('MVP Model MSE = ' + f'{score_MSE:.4f}' + ' | Model MAE = ' + f'{score_MAE:.4f}' + ' (units are rating points)')

# %% Fun Time!

coolMovies = ['Hackers', 'Sneakers', 'Avatar', 'Cool Hand Luke', 'Robocop',
              'The 5th Element', 'Alien', 'Aliens', 'Cool Runnings', 'Beetlejuice',
              'Austin Powers: The Spy Who Shagged Me', 'Memento', 'I Heart Huckabees',
              '50 First Dates', 'Justice League', 'Mars Needs Moms', 
              'Sinbad: Legend of the Seven Seas', 'Gigli', 'Titan A.E.', 'Monster Trucks',
              '47 Ronin', 'Wet Hot American Summer', 'Crash', 'Kill Bill', 'Kill Bill 2',
              'Snakes on a Plane', 'Machete', 'Nacho Libre', 'Spice World'] #some of these are cool, and some are in because they were not at all cool... can you guess which are which?

for movie in coolMovies:
    if np.sum(Z == movie) > 0:
        movieLoc = np.where(Z == movie)[0][0]
        moviePredictedRating = model.predict( np.array( [X[movieLoc],] ))[0][0] * 4.5 + 5.5 #convert to units of rating points
        movieEstablishedRating = Y[movieLoc] * 4.5 + 5.5                                    #convert to units of rating points
        print('Movie: "' + movie + '"	| Predicted Rating=' + f'{moviePredictedRating:.2f}' + '	| IMDb Rating=' + str(movieEstablishedRating))
    else:
        print('Movie not found: "' + movie + '"')




