# Machine Learning Classification Algorithms

This repository provides from-scratch implementation of popular Machine Learning 
algorithms used for classification. The main purpose is to provide insight, along with 
in-code comments on the classifiers steps.

## Classifiers

The following is a list of currently implemented classifiers (though most of them 
need some refactoring for improved readability):

* Neural Network
* K-Nearest Neighbor
* Naive Bayes
* AdaBoosting (SAMME algorithm by Stanford)

### Pending for implementation

* Logistic Regression
* Decision Trees
* Random Forest

## Classifiers API
For consistency purposes each classifier will follow the same interface, including a 
fit and a predict method. Although each classifier may require different parameters for
initialization depending on the specific classifier.
