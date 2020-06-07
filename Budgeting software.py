import pandas as pd
from sklearn import tree
import pydotplus
from sklearn.tree import DecisionTreeClassifier
import matplotlib.pyplot as plt
import matplotlib.image as pltimg
import numpy as np
import datetime
import nltk
import re
from IPython.display import Image

#imports

from sklearn.tree import DecisionTreeClassifier # Import Decision Tree Classifier
from sklearn.model_selection import train_test_split # Import train_test_split function
from sklearn import metrics #Import scikit-learn metrics module for accuracy calculation
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.feature_extraction.text import TfidfVectorizer
from datetime import datetime
from sklearn.naive_bayes import MultinomialNB
from string import digits

#more funky imports



df = pd.read_csv (r'C:\Users\E\Downloads\statement.csv', names = ['Trans #', 'Date', 'Type', 'Withdraw', 'Deposit', 'Remaining', 'Label'])
df_CC = pd.read_excel (r'C:\Users\E\Downloads\pcbanking.xlsx', names = ['Date', 'Trans_Type', 'Amount', 'Label'])
#bring in traning data

df_CC = df_CC.drop(df_CC.loc[(df_CC['Trans_Type'].str.contains("PAYMENT - THANK YOU"))].index)
df_CC.sort_values('Date')


#print(df_CC['Date'])
df_CC['Date'] = df_CC['Date'].dt.strftime("%a %d %b %Y")
df_CC['Date'] = df_CC['Date'].str.strip()

df_CC['Trans_Type'] = df_CC['Trans_Type'].str.strip()



d = {'Gas':0, 'Gifts': 1, 'Food':2, 'School': 3, 'House':4, 'Dates':5, 'Work':6, 'Trips':7, 'Online':8, 'Clothes':9 , 'Projects':10}
df_CC['Label'] = df_CC['Label'].map(d)


#print(df_CC['Date'])
#filter CC data and sort by date


purchase_df = df.loc[(df['Type'].str.contains("Purchase"))]
pd.to_datetime(purchase_df['Date'])
purchase_df.sort_values('Date')

deposits_1 = df.loc[(df['Type'].str.contains("ATM Deposit"))]

deposits_2 =  df.loc[(df['Type'].str.strip() == "Deposit")]

CC_Payment_df = df.loc[(df['Type'].str.contains("Payment"))]
pd.to_datetime(CC_Payment_df['Date'])
CC_Payment_df.sort_values('Date')

salary_df = df.loc[(df['Type'].str.contains("Payroll"))]
pd.to_datetime(salary_df['Date'])
salary_df.sort_values('Date')

interest_df = df.loc[(df['Type'].str.contains("Interest"))]
pd.to_datetime(interest_df['Date'])
interest_df.sort_values('Date')

deposit_df = pd.concat([deposits_1, deposits_2])
deposit_df["Date"] = pd.to_datetime(deposit_df.Date)
deposit_df.sort_values('Date')
#Filter data from bank account, we want to use the ML structure on the purchase_df

#Vectorize values
X = df_CC["Trans_Type"].str.replace('\d+', '')
y = df_CC['Label']
z = df_CC['Date'].str.replace('\d+', '')

keywords = open("keyphrases.txt", "r")
keywords = keywords.read().split()

keyDates = ['Mon', 'Tue', 'Wed', 'Thu', 'Fri', 'Sat', 'Sun', 'Jan', 'Feb', 'Mar', 'Apr', 'May']

cv = TfidfVectorizer(stop_words = 'english')

cv2 = TfidfVectorizer(stop_words = 'english')


x_traincv = cv.fit_transform(keywords)

x_traincv = cv.fit_transform(X)

z_traincv = cv2.fit_transform(keyDates)
z_traincv = cv2.fit_transform(z)


x_names = cv.get_feature_names()
z_names = cv2.get_feature_names()
a_names = ["Amount"]


x_vectors = pd.DataFrame(x_traincv.toarray(), columns= x_names)
z_vectors = pd.DataFrame(z_traincv.toarray(), columns= z_names)
df_CC['Amount'] = abs(df_CC['Amount'])
df_CC = df_CC.reset_index()


x_train = pd.concat([x_vectors, z_vectors, df_CC['Amount']], axis = 1)



X_train, X_test, y_train, y_test = train_test_split(x_train, y, test_size=0.1, random_state=1) # 70% training and 30% test


clf = DecisionTreeClassifier(min_samples_split = 2)

df = clf.fit(X_train, y_train)

y_pred = clf.predict(X_test)

y_pred  = pd.DataFrame(y_pred)


print("Accuracy:",metrics.accuracy_score(y_test, y_pred))

results = metrics.accuracy_score(y_test,y_pred)



classes = ['Gas', 'Gifts', 'Food', 'School', 'House', 'Dates', 'Work', 'Trips', 'Online', 'Clothes', 'Projects']

from sklearn import tree

dot_data = tree.export_graphviz(clf , feature_names = x_names + z_names + a_names, class_names = classes)
graph = pydotplus.graph_from_dot_data(dot_data)

graph.write_png('Tree.png')
