# CART Based Budgeting Software

## About

Code and other bits for a Classification and Regresstion Tree based budgeting software based on [this article](http://www.pinchofintelligence.com/managing-budget-excel-machine-learning/).

## Necessary Python Dependencies

* pandas
* sklearn
* pydotplus
* numpy
* nltk
* tkinter
* dateutil
   
## Tuning Figures  

The following figures were captured to show how the tree performed with adjustments to various tuning hyperparameters. 
  
* Max_features.png -> accuracy of CART Train/Test/Split (25% training) over all avaliable max_features (around 200 based on my training data). Train/Test/Split Iterated randomly
* Min_samples_split.png -> accuracy of CART Train/Test/Split (25% training) from 1 to 300 min_samples_split perameters. Train/Test/Split Iterated randomly
* Min_samples_leaf.png -> accuracy of CART Train/Test/Split (25% training) from 1 to 300 min_samples_leaf perameters. Train/Test/Split Iterated randomly
* Tree_depth_graph.png -> Tree including Purchase descrition and amount
* Tree_depth_graph2.png -> Tree including Purchase descrition and amount and date
