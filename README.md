# ml-star-classification

Machine Learning Project - Star Type Classification

**Problem:** the main purpose of the project is to successfully classify data found the dataset, which consists
star data with different features, since there are more than two classes (6 in total)
, the problem becomes a multi-class classification problem.

The package itself includes 4 types of classifiers, as well as a base classifier which only acts as an abstract class for the rest of them:
* Neural Network
* K-Nearest Neighbor
* Naive Bayes
* AdaBoosting (SAMME algorithm by Stanford)

Each classifier contains a fit, predict and evaluate functions, which are already in the code

**How to use:**
You can run the program on the main.py file, form here everything is already imported. The 4 classifiers used are already commented, they just need to be uncommented (one at a time).

The parameter cross_validation tells if K-fold cross validation will be run or just a single instance of the classifier

To obtain the results just run each classifier separately under k-fold, that will print the average of the metrics of the training folds, as well as the final metrics for the test set.

In order to turn off the data normalization, just comment out lines 20 to 23, and you will see how in most cases the results are lower for all classifiers.

GitHub repo:
https://github.com/JosephS96/ml-star-classification