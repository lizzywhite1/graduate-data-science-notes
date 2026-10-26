## Step 1: Clarify the problem and constraints 
Frame the business problem correctly, before beginning with solutions.

Begin by framing the objective. **What are we trying to model? What is the dependent variable?**

Additional consdierations:
1. How has this problem been solved in the past? Is there a **baseline model** to compare our results against?
2. Is ML even needed? Is it feasible to apply ML to this problem (will we face legal or ethical constraints?). Can we break it down into sub-problems, does every sub-problem need ML?
3. What is the cost of an incorrect prediction? Are false positives worse than false negatives? 

Technical considerations involve considerations regarding latency, volume of input recieved per unit measurment, where and how is this model deployed?

## Step 2: Establish metrics
- Pick simple, observable and attributable metrics. This is usually the business KPIs (e.g. time to resolve customer request).
- Align model performance metrics with business KPIs. 
- Choose a single metric rather than multiple as this allows for comparasion of models.
- If you wanted to combine metrics there are multiple approaches
    - You have a chain of requirements e.g. for classification model you could require we fix recall at 95%, whilst optimising for precision. 
    - Blend multiple metrics in a weight calculation or use a pre-defined weightted metric e.g. F1 Score.
    - Our final metric is known as the Overall Evaluation Criterion (OEC)
- Establish **what does success look like**? What threshold for this OEC is good enough? This is why enquiring about a baseline is useful. 


## Step 3: Understanding the data sources
- Remember, *garbage in garbage out*, we require good data to build a good model.
- What data would we ideally need, what do we have? Is it feasible to obtain this? 
- Ask questions to understand the data.
    - How recent is this data?
    - How much data do we have, will this be enough?
    - How often will it be updated?
    - How was the data collected? Is there any *sampling, selection or response bias*?

## Step 4: Explore the data 
Important ways to explore the data. 
- Obtain summary statistics about each feature. Features with minimal variance will have little descriptive power. Which columns have a lot of variation. Which have a lot of missing values? 
- Look at summary statistics for each feature. Mean, median, mode and quantiles. 
- Visualise the distributions of features of interest to look for skewness (assymetry of distribution) and kurtosis (measures weight and extremity of tails).
- Boxplots for range and quantiles for continuous features. Group by categories. 
- Correlation matrix to inspect linear relationships between variables, or use scatter plots to look for non-linear relationships.

## Step 5: Clean the data
1. Check for duplicate rows or columns 
2. Check for incorrect values that don't match schema 
3. To handle the missing data:
    - Understand the reason behind it being missing
    - You could impute based on mean / median 
    - Use a model or distribution to impute the missing data 
    - Drop rows with mising values as a last resort
4. Deal with outliers, NEVER drop outliers just for the sake of it. Should only be dropped if it is clear it should not be part of the dataset. If it is a data entry issue is there a clear fix? For example, someone could be 4ft 2 but it's clear an adult who is 0.2 metres is not true, maybe it was meant to be 2 metres?

## Step 6: Feature Engineering 
Feature engineering is composed of feature selection and feature preprocessing. See page feature engineering for more details. Feature selection and preprocessing are important for model performance. 

## Step 7: Model Selection
We now have the business constraints, evaluation metrics and data sources, we should now decide what is the best model. 

Once you have narrowed down your search to models for unsupervised/supervised data and whether it is a classification or regression task, things to consider include
1. Training & Prediction Speed
2. Computational Budget: for example, NNs are more intensive to train than 
3. Volume & dimensionality of data: can our model handle the dimensionality of the data and the amount of observations
4. Categorical vs Numerical Features: linear regression must use dummy variables to handle categorical variables whereas trees handle it directly. 
5. Explainability: interable models like linear regression may be favourable if working in an enviroment with strict regulations. 

**Choosing the right ML model**

## Step 8: Model Training & Evaluation
When training the model(s) consider
1. Train-validation-test split
2. Cross validation
3. Hyperparameter tuning 

Consider how to compare models, how the parameters are learned, how you would know the performance is good enough to ship the model and how to avoid overfitting. Edge cases to consider may involve how you handle biased trsaining data, assess feature importance, or deal with data imbalances. 

If a model is taking ages to train, it is worth investigating various sampling techniques. If dataset is imbalanced, then undersampling or oversampling methods may be used. 

## Step 9: Deployment 
## Step 10: Iterate