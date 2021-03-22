from __future__ import absolute_import, division, print_function, unicode_literals

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from IPython.display import clear_output
from six.moves import urllib
import csv
import yfinance as yf

import tensorflow.compat.v2.feature_column as fc

import tensorflow as tf

CSV_COLUMN_NAMES = ['x'+str(i) for i in range(1, 16)]
CSV_COLUMN_NAMES.append('y')
categories = ['down', 'up']

#train_path = tf.keras.utils.get_file(
#    "iris_training.csv", "https://storage.googleapis.com/download.tensorflow.org/data/iris_training.csv")
#test_path = tf.keras.utils.get_file(
#    "iris_test.csv", "https://storage.googleapis.com/download.tensorflow.org/data/iris_test.csv")
f = open('Stock data.csv', 'r')

allData = pd.read_csv(f, names=CSV_COLUMN_NAMES, header=0)
train = allData.iloc[1:4500,:]
test = allData.iloc[4501:4750,:]

train_y = train.pop('y')
test_y = test.pop('y')


def input_fn(features, labels, training=True, batch_size=256):
    # Convert the inputs to a Dataset.
    dataset = tf.data.Dataset.from_tensor_slices((dict(features), labels))

    # Shuffle and repeat if you are in training mode.
    if training:
        dataset = dataset.shuffle(1000).repeat()
    
    return dataset.batch(batch_size)

my_feature_columns = []
for key in train.keys():
    my_feature_columns.append(tf.feature_column.numeric_column(key=key))
#print(my_feature_columns)


classifier = tf.estimator.DNNClassifier(
    feature_columns=my_feature_columns,
    # Two hidden layers of 30 and 10 nodes respectively.
    hidden_units=[15, 10],
    # The model must choose between 3 classes.
    n_classes=2)

classifier.train(
    input_fn=lambda: input_fn(train, train_y, training=True),
    steps=5000)

eval_result = classifier.evaluate(
    input_fn=lambda: input_fn(test, test_y, training=False))
print(eval_result)
print('\nTest set accuracy: {accuracy:0.3f}\n'.format(**eval_result))



def input_fn(features, batch_size=1):
    # Convert the inputs to a Dataset without labels.
    return tf.data.Dataset.from_tensor_slices(dict(features)).batch(batch_size)

features = ['x'+str(i) for i in range(1, 16)]


#Find amount of money made in 1 month
profitsArray = []
tickerList = []
g = open('Stock cross validation set.csv', 'r')
crossValData = pd.read_csv(g, names=CSV_COLUMN_NAMES, header=0)
print(crossValData)
allProbabilities = []
for i in range(0, 503):
    predict = {}
    #i is the stock in the table
    for j in range(0, 15):
        predict[features[j]] = [float(crossValData.iloc[i, j+1])]
    predictions = classifier.predict(input_fn=lambda: input_fn(predict))
        
    for pred_dict in predictions:
        class_id = pred_dict['class_ids'][0]
        probability = pred_dict['probabilities'][class_id]
        allProbabilities.append(probability)
        ticker = crossValData.index[i]
        print(ticker)
        print('Prediction is "{}" ({:.1f}%)'.format(
            categories[class_id], 100 * probability))

        #Calculate profit made from $100 investment
        
        tickerList.append(ticker)
        yfTicker = yf.Ticker(ticker)
        hist = yfTicker.history(start='2021-02-15', end='2021-03-13')
        buyPrice = hist.iloc[0,3]
        sellPrice = hist.iloc[hist.shape[0]-1,3]
        profit = (((sellPrice-buyPrice)/buyPrice))*100
        profitsArray.append(profit)
        
#pd.DataFrame(allProbabilities).hist(bins=20)
plt.scatter(allProbabilities,profitsArray)

#Print top 20 stock picks and results if we would've invested in them
#We have list of 503 tickers, 503 prediction %s, 503 profit amounts
stockPicks = pd.DataFrame({'tick':tickerList,'certainty':allProbabilities,'profit':profitsArray})
stockPicks = stockPicks.sort_values('certainty', ascending=False)
print(stockPicks.head(20))

print("Top 5 average: " + str(stockPicks.cumsum().iloc[4, 2]/5))
print("Top 10 average: " + str(stockPicks.cumsum().iloc[9, 2]/10))
print("Top 20 average: " + str(stockPicks.cumsum().iloc[19, 2]/20))
print("Market average: " + str(stockPicks.cumsum().iloc[502, 2]/503))


tickerList = []
g = open('Current stock data.csv', 'r')
CSV_COLUMN_NAMES.pop()
crossValData = pd.read_csv(g, names=CSV_COLUMN_NAMES, header=0)
allProbabilities = []
for i in range(0, 503):
    predict = {}
    #i is the stock in the table
    for j in range(0, 15):
        predict[features[j]] = [float(crossValData.iloc[i, j])]
    predictions = classifier.predict(input_fn=lambda: input_fn(predict))
        
    for pred_dict in predictions:
        class_id = pred_dict['class_ids'][0]
        probability = pred_dict['probabilities'][class_id]
        allProbabilities.append(probability)
        
        print('Prediction is "{}" ({:.1f}%)'.format(
            categories[class_id], 100 * probability))

        #Calculate profit made from $100 investment
        ticker = crossValData.index[i]
        tickerList.append(ticker)


stockPicks = pd.DataFrame({'tick':tickerList,'certainty':allProbabilities})
stockPicks = stockPicks.sort_values('certainty', ascending=False)
print(stockPicks.head(20))




