# CART Based Budgeting Software

A few months ago I was thinking of ways to improve my workflow while budgeting until I came across this  http://www.pinchofintelligence.com/managing-budget-excel-machine-learning/ article. I took inspiration and used it as a way to learn the basics of python, pandas and Scikit-Learn's ML functionality. I tried using both the Bayes-Multinomial Classification meathod and Decision Tree (CART) but found the Tree worked very well with almost no tuning (although I took the time to improve its performance even further).

Listed files:

CART_Budget -> ML and processing script
Max_features.png -> accuracy of CART Train/Test/Split (25% training) over all avaliable max_features (around 200 based on my training data). Train/Test/Split Iterated randomly
Min_samples_split.png -> accuracy of CART Train/Test/Split (25% training) from 1 to 300 min_samples_split perameters. Train/Test/Split Iterated randomly
Min_samples_leaf.png -> accuracy of CART Train/Test/Split (25% training) from 1 to 300 min_samples_leaf perameters. Train/Test/Split Iterated randomly
Tree.png -> Final budgeting CART tree
Tree_depth_graph.png -> Tree including Purchase descrition and amount
Tree_depth_graph.png -> Tree including Purchase descrition and amount and date
dict.txt -> dictionary of possible fields for algorithm to classify to
keyphrases.txt ->depreciated corpus of countvectorizor



