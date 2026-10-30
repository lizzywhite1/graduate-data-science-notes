# Feature Engineering & Selection 

### What is feature engineering and selection? Why is it important?

Feature engineering preprocesses data into relevant information for machine learning models. Using high-quality, refined inputs improves model performance and enhances model interpreability. Feature selection prevents overfitting and computational cost and improves generalisation. 

## Data Cleaning Steps 
### 1. Missing Data
Missing values can lead to poor model performance, or some models won't run with missing values. 

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
1. 



### 2. Handling and looking for outliers (vs. rare events)



## Feature Engineering 
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




This is a short data cleaning checklist I follow 

1. Are there any obvious columns that should be dropped?
Obvious examples include (usually) a unique ID, and categorical variables with only one value. Use domain knowledge and common sense to identify if there are any others which are very obviously not useful. 

```python
df.describe(include = 'all') # will include summary stats of all dtype columns

df = df.drop(columns = ['id', 'useless_column'], axis = 1)
```


2. Check for duplicates 

```python
df[df.duplicated()]
```

3. Check and handle null values. 

- Is there a reason for it being null? Is it associated with another categorical feature?
If NaN just means None literally may be worthwhile filling as None
```python
df.isna().sum()

df['col'].fillna('None')
```

4. Check for incorrect datatypes
- Make sure stings of numbers are integers
- Make sure if n_unique == 2 we change to bool
- Change categories to category data type 

```python
df.info(verbose = 'True') # prints dtypes of columns

# to change one column to numeric
df['column_name' = df['column_name'].apply(lambda x: pd.to_numeric(x, errors = 'ignore'))

# To change str to category 
def str_to_cat(df):
    cols = [col for col in df.select_dtypes(include='str') if df[col].nunique() <= 5]
    return df.astype({col: 'category' for col in cols})

df = df.pipe(str_to_cat)

# Or mapping strs to bools 
def yes_no_cols(df):
    cols = [col for col in df.columns if df[col].nunique() == 2 and df[col].dtype == 'str' and any(x in ['Yes', 'No'] for x in df[col].unique().tolist())]
    return df.assign(**{col: df[col].map({'Yes': True, 'No': False}) for col in cols})

df = df.pipe(yes_no_cols)
```





Transformations (log, capping, flooring) , binning, dimensionality reduction
One hot encoding hashing 
When is standardisation needed 
Feature selection methods 
NLP (stemming, )