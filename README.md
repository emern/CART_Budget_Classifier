# CART Based Budgeting Software

A few months ago I was thinking of ways to improve my workflow while budgeting until I came across this  http://www.pinchofintelligence.com/managing-budget-excel-machine-learning/ article. I took inspiration and used it as a way to learn the basics of python, pandas and Scikit-Learn's ML functionality. I tried using both the Bayes-Multinomial Classification methods and Decision Tree (CART) but found the Tree worked very well with almost no tuning (although I took the time to improve its performance even further).  
  
Listed files:  
  
CART_Budget -> ML and processing script </br>
Max_features.png -> accuracy of CART Train/Test/Split (25% training) over all avaliable max_features (around 200 based on my training data). Train/Test/Split Iterated randomly</br>
Min_samples_split.png -> accuracy of CART Train/Test/Split (25% training) from 1 to 300 min_samples_split perameters. Train/Test/Split Iterated randomly</br>
Min_samples_leaf.png -> accuracy of CART Train/Test/Split (25% training) from 1 to 300 min_samples_leaf perameters. Train/Test/Split Iterated randomly</br>
Tree.png -> Final budgeting CART tree</br>
Tree_depth_graph.png -> Tree including Purchase descrition and amount</br>  
Tree_depth_graph.png -> Tree including Purchase descrition and amount and date</br>  
dict.txt -> dictionary of possible fields for algorithm to classify to</br>
keyphrases.txt ->depreciated corpus of countvectorizor</br>


