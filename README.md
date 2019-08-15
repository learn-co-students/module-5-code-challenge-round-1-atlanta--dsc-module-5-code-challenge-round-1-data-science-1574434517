
# Module 5 Assessment 

Welcome to your Module 5 Assessment. You will be tested for your understanding of concepts and ability to programmatically solve problems that have been covered in class and in the curriculum. 

**_Read the instructions very carefully!_** You will be asked both to write code and respond to a few short answer questions.  

The goal here is to demonstrate your knowledge. Showing that you know things about certain concepts and how to apply different methods is more important than getting the best model.

You will have up to 120 minutes to complete this assessment.

The sections of the assessment are:

- Decision Trees
- Ensemble Models 
- PCA
- Clustering

**Note on the short answer questions**: 
> Please use your own words, even if you consult another source to help you craft your response. Short answer questions are not necessarily being assessed on grammatical correctness or sentence structure, but do your best to communicate your answers clearly!


## Decision Trees

### Concepts 
You're given a dataset of 30 elements, 15 of which belong to a positive class (denoted by **+**) and 15 of which do not (denoted by **-**). These elements are described by two attributes, A and B, that can each have either one of two values, true or false. 

The diagrams below show the result of splitting the dataset by attribute: the diagram on the left hand side shows the class distribution of elements obtained when splitting the data by attribute A, and the diagram of the right hand side shows the class distribution of elements obtained when splitting the data by attribute B. 

<img src="images/decision_stump.png">

**1.1) Which one of the two attributes resulted in the best split of the original data? How do you select the best attribute to split a tree at each node?** _(Hint: Mention splitting criteria)_


```python
# Your answer here 
```

### Decision Trees for Regression 

In this section, you will use decision trees to fit a regression model to the Combined Cycle Power Plant dataset. 

This dataset is from the UCI ML Dataset Repository, and has been included in the `data` folder of this repository as an Excel `.xlsx` file, `Folds5x2_pp.xlsx`. 

The features of this dataset consist of hourly average ambient variables taken from various sensors located around a power plant that record the ambient variables every second.  
- Temperature (AT) 
- Ambient Pressure (AP) 
- Relative Humidity (RH)
- Exhaust Vacuum (V) 

The target to predict is the net hourly electrical energy output (PE). 

The features and target variables are not normalized.

In the cells below, we import `pandas` and `numpy` for you, and we load the data into a pandas DataFrame. We also include code to inspect the first five rows and get the shape of the DataFrame.


```python
import pandas as pd 
import numpy as np 

# Load the data
filename = 'data/Folds5x2_pp.xlsx'
df = pd.read_excel(filename)
```


```python
# Inspect the first five rows of the dataframe
df.head()
```


```python
# Get the shape of the dataframe 
df.shape
```

Before fitting any models, you need to create training and testing splits for the data.

Below, we split the data into features and target ('PE') for you. 


```python
X = df[df.columns.difference(['PE'])]
y = df['PE']
```

**1.2) Split the data into training and test sets. Create training and test sets with `test_size=0.5` and `random_state=0.1`.** 


```python
# Your code here. Replace None with appropriate code. 

X_train, X_test, y_train, y_test = None
```

**1.3) Fit a vanilla decision tree regression model with scikit-learn to the training data.** Set `random_state = 1` for reproducibility. **Evaluate the model on the test data.** 


```python
# Your code here 
```

**1.4) Obtain the mean squared error, mean absolute error, and coefficient of determination (r2 score) of the predictions on the test set.** _Hint: Look at the `sklearn.metrics` module._


```python
# Your code here. Replace None with appropriate code. 

print("Mean Squared Error:", None)
print("Mean Absolute Error:", None)
print("R-squared:", None)
```

### Hyperparameter Tuning of Decision Trees for Regression

Next, you'll find the optimal `max_depth`, `min_samples_leaf`, and `min_samples_split` hyperparameters for a decision tree regression model. You'll use the same data as in the previous question.

In the cell below, we have provided you with a function, `evaluate_performance_with_different_hyperparameters`. 


```python
def evaluate_performance_with_different_hyperparameters(hyperparameter_name, hyperparameter_values):
    
    mses_test = []
    
    for value in hyperparameter_values:
        if hyperparameter_name == 'max_depth':
            dt = DecisionTreeRegressor(max_depth=value, random_state=1)
        elif hyperparameter_name == 'min_samples_split':
            dt = DecisionTreeRegressor(min_samples_split=value, random_state=1)
        else:
            dt = DecisionTreeRegressor(min_samples_leaf=value, random_state=1)
        dt.fit(X_train, y_train)
        y_pred = dt.predict(X_test)
        mse_test = mean_squared_error(y_test, y_pred)
        mses_test.append(mse_test)
        
    return mses_test
```

This function loops through all possible hyperparameter values in the array `hyperparameter_values`, fits a model with each one of hyperparameter values using the training dataset, and evaluates the trained model performance on the test data. It uses the mean squared error metric to evaluate model performance. 
* Note that you only need to pass the hyperparameter name as a string and a list of hyperparameter values to this function to obtain a list of mean squared error scores for each of the models trained using the different hyperparameter values. 

After you have obtained model performance on test data, you will select the best `max_depth`, `min_samples_leaf`, and `min_samples_split`, train a new model using the best of these hyperparameters, and evaluate model performance. 

In the cell below, we create three variables for you, one for each hyperparameter you'll tune, that contain the hyperparameter values we want to train our models with. 


```python
# max_depth
max_depths = np.arange(1, 21)

# min_samples_split
min_samples_splits = np.arange(5, 100, 5)

# min_samples_leaf
min_samples_leafs = np.arange(5, 100, 5)
```

Next, you'll obtain model performance on the test dataset for models fit to the training data using the different hyperparameters.

**1.5) Create three variables, one for each hyperparameter you'll tune, that contain a list of the mean squared errors obtained by evaluating the trained models on the test data as hyperparameter values are changed.** 

_Hint: Set these variables equal to the result of the function call, for the hyperparameters `max_depth`, `min_samples_split`, `min_samples_leaf` respectively._


```python
# Replace None with appropriate code

mses_test_depths = None
mses_test_splits = None 
mses_test_leafs = None 
```

Now that you have evaluated model performance using different hyperparameters, you will select the best hyperparameter to use and train a new model using these hyperparameters.

We have provided you with a function `get_best_param` in the cell below so you can get the best hyperparameter of interest given a list of mean squared errors and a list of hyperparameter values. 


```python
def get_best_param(mse_scores, hyperparameter_values):
    # get index of minimum mse scores 
    ix = np.argmin(mse_scores)
    return hyperparameter_values[ix]
```

**1.6) Find the best hyperparameters and print them out.** 
* Place the best `max_depth` in a variable called `best_max_depth`.
* Place the best `min_samples_split` in a variable called `best_min_samples_split`.
* Place the best `min_samples_leaf` in a variable called `best_min_samples_leaf`.


```python
# Replace None with appropriate code

best_max_depth = None
best_min_samples_split = None
best_min_samples_leaf = None

print("The best max_depth found is:", best_max_depth)
print("The best min_samples_split found is:", best_min_samples_split)
print("The best min_samples_leaf found is:", best_min_samples_leaf)
```

**1.7) Use the optimal hyperparameters to fit a decision tree regression model to the training data.** Call the model `dt_tuned`.  Use `random_state = 1` for reproducibility.


```python
# Replace None with appropriate code 

dt_tuned = None

dt_tuned.fit(None, None)
```

**1.8) Evaluate the model's performance (mean squared error, mean absolute error, and coefficient of determination) on the test data. Did tuning the model hyperparameters improve model performance on the test dataset compared to the vanilla decision tree you trained?** 


```python
# Replace None with appropriate code 

y_pred_tuned = dt_tuned.predict(None)

print("Mean Squared Error:", None)
print("Mean Absolute Error:", None)
print("R-squared:", None)
```


```python
# Your answer here
```

## Ensemble Methods

### Introduction to Ensemble Methods

**2.1) Explain how the random forest algorithm works. Why are random forests resilient to overfitting?**

_Hint: Your answer should discuss bagging and the subspace sampling method._


```python
# Your answer here
```

### Random Forests and Hyperparameter Tuning using GridSearchCV

In this section, you will perform hyperparameter tuning for a Random Forest classifier using GridSearchCV. You will use `scikit-learn`'s wine dataset to classify wines into one of three different classes. 

After finding the best estimator, you will interpret the best model's feature importances. 

In the cells below, we have loaded the relevant imports and the wine data for you. 


```python
# Relevant imports 
from sklearn.datasets import load_wine

# Load the data 
wine = load_wine()
X, y = load_wine(return_X_y=True)
X = pd.DataFrame(X, columns=wine.feature_names)
y = pd.Series(y)
y.name = 'target'
df = pd.concat([X, y.to_frame()], axis=1)
```

In the cells below, we inspect the first five rows of the dataframe and compute the dataframe's shape.


```python
df.head()
```


```python
df.shape
```

We also get descriptive statistics for the dataset features, and obtain the distribution of classes in the dataset. 


```python
X.describe()
```


```python
y.value_counts().sort_index()
```

You will now perform hyper-parameter tuning for a Random Forest classifier.

**2.2) Use the following parameters to construct the `param_grid` dictionary to pass to `GridSearchCV` when instantiating the object.** 

* `n_estimators`: 5, 10, 20, and 30
* `max_features`: 'auto', 'sqrt', 'log2'
* `max_depth`: 4, 5, and 6
* `criterion`: 'gini' and 'entropy' 


```python
# Replace None with relevant code 
param_grid = None
```

Now that you have created the `param_grid` dictionary of hyperparameters, let's continue performing hyperparameter optimization of a Random Forest Classifier. 

In the cell below, we include the relevant imports for you.


```python
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import GridSearchCV
```

**2.3) Create an instance of a Random Forest classifier estimator; call it `rfc`.** Make sure to set `random_state=42` for reproducibility. 


```python
# Replace None with appropriate code
rfc = None
```

**2.4) Create an instance of an `GridSearchCV` object and fit it to the data.** Call the instance `cv_rfc`. 

* Use the random forest classification estimator you instantiated in the cell above, the parameter grid dictionary constructed, and make sure to perform 5-fold cross validation. 
* The fitting process should take 10 - 15 seconds to complete. 


```python
# Replace None with appropriate code 
cv_rfc = None 

cv_rfc.fit(None, None)
```

**2.5) What are the best training parameters found by GridSearchCV?** 

_Hint: Explore the documentation for GridSearchCV._ 


```python
# Replace None with appropriate code 
None 
```

In the cell below, we create a variable `best_model` that holds the best model found by the grid search.


```python
best_model = cv_rfc.best_estimator_
```

Next, we give you a function that creates a horizontal bar plot to visualize the feature importances of a model, sorted in descending order. 


```python
import matplotlib.pyplot as plt 
%matplotlib inline 

def create_plot_of_feature_importances(model, X):
    ''' 
    Inputs: 
    
    model: A trained ensemble model instance
    X: a dataframe of the features used to train the model
    '''
    
    feat_importances = model.feature_importances_

    features_and_importances = zip(X.columns, feat_importances)
    features_and_importances = sorted(features_and_importances, 
                                     key = lambda x: x[1], reverse=True)
    
    features = [i[0] for i in features_and_importances]
    importances = [i[1] for i in features_and_importances]
    
    plt.figure(figsize=(10, 6))
    plt.barh(features, importances)
    plt.gca().invert_yaxis()
    plt.title('Feature Importances')
    plt.xlabel('importance')
```

**2.6) Create a plot of the best model's feature importances. What are this model's top 3 features in order of descending importance?** 

_Hint: To create the plot, pass the appropriate parameters to the function above._


```python
# Your code here.
```


```python
# Your answer here 
```

## Principal Components Analysis

### Training a model with PCA-extracted features

In this section, you'll apply the unsupervised learning technique of Principal Components Analysis to the wine dataset. 

You'll use the principal components of the dataset as features in a machine learning model. You'll use the extracted features to train a vanilla Random Forest Classifier, and compare model performance to a model trained without PCA-extracted features. 

In the cell below, we import the data for you, and we split the data into training and test sets. 


```python
from sklearn.datasets import load_wine
X, y = load_wine(return_X_y=True)

wine = load_wine()
X = pd.DataFrame(X, columns=wine.feature_names)
y = pd.Series(y)
y.name = 'class'

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=42)
```

**3.1) Fit PCA to the training data.** 

Call the PCA instance you'll create `wine_pca`. Set `n_components=0.9` and make sure to use `random_state = 42`.

_Hint: Make sure to include necessary imports for preprocessing the data!_


```python
# Your code here 
```

**3.2) What is the meaning of setting `n_components` to 0.9?** 

_Hint: Look at the documentation for `PCA` in scikit-learn._


```python
# Your answer here
```

**3.3) How many principal components are there in the fitted PCA object?**

_Hint: Look at the list of attributes of trained `PCA` objects in the scikit-learn documentation_


```python
# Replace None with appropriate code 
print(None)
```

Next, you'll reduce the dimensionality of the training data to the number of components that explain at least 90% of the variance in the data, and then you'll use this transformed data to fit a Random Forest classification model. 

You'll compare the performance of the model trained on the PCA-extracted features to the performance of a model trained using all features without feature extraction.

**3.4) Transform the training features into an array of reduced dimensionality using the `wine_pca` PCA object you've fit in the previous cell.** Call this array `X_train_pca`.


```python
# Replace None with appropriate code  
X_train_pca = None
```

Next, we create a dataframe from this array of transformed features and we inspect the first five rows of the dataframe for you. 


```python
# Create a dataframe from this array of transformed features 
X_train_pca = pd.DataFrame(X_train_pca)

# Inspect the first five rows of the transformed features dataset 
X_train_pca.head()
```

You will now use the PCA-extracted features to train a random forest classification model.

**3.5) Instantiate a vanilla Random Forest Classifier (call it `rfc`) and fit it to the transformed training data.** Set `random_state = 42`. 


```python
# Replace None with appropriate code 
rfc = None
rfc.fit(None, None)
```

**3.6) Evaluate model performance on the test data and place model predictions in a variable called `y_pca_pred`.**

_Hint: Make sure to transform the test data the same way as you transformed the training data!!!_


```python
# Your code here 
```

In the cell below, we print the classification report for the model performance on the test data. 


```python
from sklearn.metrics import classification_report
print(classification_report(y_test, y_pca_pred))
```

Run the cell below to fit a vanilla Random Forest Classifier to the untransformed training data,  evaluate its performance on the untransformed test data, and print the classification report for the model. 


```python
vanilla_rfc = RandomForestClassifier(random_state=42)
vanilla_rfc.fit(X_train, y_train)

y_pred = vanilla_rfc.predict(X_test)

print(classification_report(y_test, y_pred))
```

**3.7) Compare model performance. Did the overall accuracy of the model improve when using the transformed features?**


```python
# Your answer here 
```

## Clustering 

### Clustering Algorithms: k-means and hierarchical agglomerative clustering

**4.1) Using the gif below for reference, describe the steps of the k-means clustering algorithm. Compare to the way clusters are created using hierarchical agglomerative clustering.** 
* If the gif doesn't run, you may access it via [this link](https://github.com/learn-co-curriculum/dsc-module-5-assessment/blob/master/images/good-centroid-start.gif).

<img src='images/good-centroid-start.gif'>


```python
# Your answer here
```

### k-means Clustering

For this question, you will apply k-means clustering to your now friend, the wine dataset. 

You will use scikit-learn to fit k-means clustering models, and you will determine the optimal number of clusters to use by looking at silhouette scores. 

We load the wine dataset for you in the cell below. 


```python
from sklearn.datasets import load_wine

X, y = load_wine(return_X_y=True)
wine = load_wine()
X = pd.DataFrame(X, columns = wine.feature_names)
```

**4.2) Write a function called `get_labels` that will find `k` clusters in a dataset of features `X`, and return the labels for each row of `X`.**

_Hint: Within the function, you'll need to:_
* _instantiate a k-means clustering model (use `random_state = 1` for reproducibility),_ 
* _fit the model to the data, and_
* _return the labels for each point._


```python
# Replace None and pass with appropriate code
def get_labels(k, X):
    
    # Instantiate a k-means clustering model with random_state=1 and n_clusters=k
    kmeans = None
    
    # Fit the model to the data
    None
    
    # return the predicted labels for each row in the data
    pass 
```

**4.3) Fit the k-means algorithm to the wine data for k values in the range 2 to 9 using the function you've written above. Obtain the silhouette scores for each trained k-means clustering model, and place the values in a list called `silhouette_scores`.** 

We have provided you with some starter code in the cell below.

_Hints: What imports do you need? Do you need to pre-process the data in any way before fitting the k-means clustering algorithm?_ 


```python
# Your code here

silhouette_scores= []

for k in range(2, 10):
    labels = None 
    
    score = silhouette_score(None, None, metric='euclidean')
    
    silhouette_scores.append(score)
```

Run the cell below to plot the silhouette scores obtained for each different value of k against k, the number of clusters we asked the algorithm to find. 


```python
plt.plot(range(2, 10), silhouette_scores, marker='o')
plt.title('Silhouette scores vs number of clusters')
plt.xlabel('k (number of clusters)')
plt.ylabel('silhouette score')
```

**4.4) Which value of k would you choose based on the plot of silhouette scores? How does this number compare to the number of classes in the wine dataset?**


```python
# Your answer here 
```
