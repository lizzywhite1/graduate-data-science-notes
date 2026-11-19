# Data Cleaning & Feature Engineering

## Data Cleaning Steps 
### 1. Missing Data
Missing values can lead to poor model performance, or some models won't run with missing values. 

Use ``df.info(verbose = True, value_counts = True)`` to get quick summary. 

**Missing Mechanisms**
- Missing Completely at Random (MCAR): no relationship between data missing and other values. Disregarding these observations would not bias model.
- Missing at Random (MAR): the probability of observation being missing depends on other entries, for example women may be less likely to disclose weight than men. Ensure you include the variables that influence misisngness in the model.
- Missing not at Random (MNAR): can depends on unobserved variables or the missing value itself, 
    - e.g. people who left study early experienced discomfort from treatment, but discomfort is not measured. If you drop these entries, the sample becomes biased. 
    - e.g. higher earning less likely to reveal earnings

Two ways to find missing mechanisms: 
1. Business Knowledge : Use logic / domain knowledge. 
2. Statistical Tests: Divide dataset into ones with / without missing values and use t-test to see if there is specific differences.

**Handling missing data** 
1. MCAR
2. MAR
3. MNAR


**Useful functions** 
```python
# Understand what rows have nulls in them 
df.isna().sum 
df.isna().sum / len(df) * 100

# Check for duplicates 
df = df.drop_duplicates()

# To drop the rows/columns 
df = df.dropna()

# To fill the rows/columns 
# To perform mean or median or mode imputation
df[col] = df[col].fillna(df[col].mean())
df[col] = df[col].fillna(df[col].mode().iloc[0])

# Forward or backward fill, useful for time series data
df[col] = df[col].fillna(method = 'ffil')
df[col] = df[col].fillna(method='bfill')

# To replace 

# Interpolation techniques
```


### 2. Handling and looking for outliers (vs. rare events)




### 3. Data Types and Mappings 

**Changing values**
- .map() uses a dictionary mapping to replace every value with specificed values, if value not specified it becomes nan
- .replace() is the same but leaves unspecified ones alone

## Feature Engineering 
### What is feature engineering? Why is it important?
Feature engineering preprocesses data into relevant information for machine learning models. Using high-quality, refined inputs improves model performance and enhances model interpreability. 
### 1. High Cardinality Labels 
Number of labels within a category is known as cardinality. High cardinality features can dominate the learning algorithm and create a lot of noise with little information. High cardinality is especially bad in tree based algorithms. 
- Check how many observations are in each category to get a better understanding if its worth keeping. 

**Ways to deal with high cardinality**
- Group categories, use business knowledge or group rare categories together 
- Group catgeories with decision tree. 
- If the labels have an order, you can turn it into ordinal data instead. 

### 2. Discretisation 
Transform continuous variables into discrete variables, by creating a set of contiguous intervals that span the range of the data's values. 

When to do it: useful when working with linear models and certain continuous features have non-linear effect on target. Useful on very noisy features. 

```python
pd.cut(df['col'], bins = range(0,101, 10)) # create bins (0,10], (10,20]...
```

### 3. Feature Scaling  
Feature scaling is a method use to standardise a collection of independent predictor variables. Usually performed during data preprocessing. 

**Why and when does feature scaling matter**
- If the range of inputs vary, for some algorithms, the objective function will not work properly. 
- Methods that utilise gradient descent is a distance based convergence algorithm which will converge faster if inputs are normalised first. 
- Additionally ML methods which involve a distance calculation, such as KNN, are affected by the magnitude of the features. Variables with larger magnitude would dominate the measure more than the smaller ones.
- Tree based algorithms are one of the few ML methods not affected by input range.

**Methods**
1. Normalisation (Z-Score Scaling): $(x-\bar x)/\sigma$. Can compress the observations in the narrow range if the feature is skewed or has has outliers, and hence affect predictive power. Works well for Gaussian features, or log transformed features.  
2. Min Max Scaling: scales each feature within a given range, usually $[0,1]$. $(x-x_{min})/(x_{max}-x_{min})$. Similarly, can compress observations in the narrow range if distribution is heavily skewed or has outliers. 
3. Robust Scaling: removes the median and scales the data according to the quantile range, defaults to IQR, $(x-\text{median})/\text{IQR}$.

### 4. Feature Scaling 

Feature encoding allows us to transform the data such that it can be processed by the ML algorithm. Types of encoding 
- **One hot encoding**: replace categorical variable or booleans with 0,1. Can expand feature space significantly if used with a category with many labels. Doesn't make the variable more predictive. 
- **Ordinal Encoding**: if a label has a clear order, replace by ordinal number. Doesn't make the variable more predictive. 
- **Count/frequency encoding**: replace each label of the categorical variable with a count / frequency within that category. E.g. turn the true/false labels of have mother, father, siblings, to number of family members. 
- **Mean Encoding**: ...
- **WOE Encoding** ...
- **Target encoding** ...

Ensure when multicollinearity is an issue, that you drop one of the $k$ one hot encodings, so you have $k-1$. Do this for any algorithm that looks at all the features at time of training, e.g. neural networks, SVM, clustering. Tree based algorithms need the entire set of binary variables to make the best split. 

It is not reccomended to use one-hot encoding for tree-based algorithms because 
- Creates weak, unbalanced splits; each split reduces impurity by a small amount. Tree has to become very deep to use the feature at all. E.g. splitting cities in UK to a one hot encoding, may be better to just have London vs Every other city, as two features. 
- Dilutes feature importance: in random forests `max_features` means excessive dummy variables also reduce expressiveness of each tree. 
- Misleading feature importance: the importance is split across various dummy variables.
- Ineffcient 


### 5. Feature Transformation





