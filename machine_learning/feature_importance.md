## Feature Selection
The process of selecting a subset of relevant features to use within a ML model building. 
- Including irrelevant or redundant features can harm model performance and cause overfitting. 

Feature selections 
- Simplifies the model making it less likely to overfit and more interperable
- Shorter training times and computational cost
- Avoids the curse of dimensionality

The type of feature selection process we use should depend on the type of model we are fitting, these two processes, feature selection and model fitting should not always be independent. 

Feature selection can be supervised or unsupervised, and model agnostic or model dependent. 

## Unsupervised Feature Selection Methods 
For unsupervised techniques 
1. Principal Component Analysis: finds new basis for data, such that each PC component maximises remaining explainable variance in data. 
2. Independent Component Analysis
3. Non Negative Matrix Factorisation 
4. t-SNE
5. Autoencoder 

For supervised techniques, general techniques are 1) Filter method 2) Wrapper methods and 3) Embedded methods. 

## Supervised Feature Selection 
### Filter Methods 
Select features based on a performance measure regardless of the ML algorithm lated deployed. They are model agnostic, less computationally expensive but usually give lower predictive performance. They do not remove multicollinearity. Evaluates features individually, lags when combination of feature improves performance of model.

Some techniques include 
1. **Variance or Mean Absolute Deviation**: remove features with low to zero variance ie. constant or quasi-constant features. Scale dependent ensure data is scaled before hand. 
2. **Correlation**: remove features that are highly correlated with one another as one will be redundant. Pearson correlation coeffcient measures the linear relationship between two variables. Spearman looks for monotonic relationships. 
    - Caveat: Will miss non-monotonic relationships like parabolic ones. Also Pearsons is influenced by outliers, spearman is not. 

3. **Information Gain/Mutal Dependence/Mutual Infomation**: How much does knowing X reduce uncertainty about Y. Zero if X and Y are independent, will detect non-monotonic/non-linear relationships. How $p$ is estimated depends on whether $X$ is discrete (contingency tables) or continuous (various methods). 

$$
I(X;Y)= \sum _{x,y} p(x,y)\log {\frac{p(x,y)}{p(x)p(y)}}
$$

4. **Chi-Squared**: compute chi-squared stat between every non-negative feature and class. Tests relationships between categorical variables. `sklearn` needs non-negative, count like features with $n\ge 5$. Intuition behind Chi-Squared test, expected count assuming independence $E = (\text{Row count} \times \text{column count})/n$, and $O$ follows roughly Poisson distribution as it's count data, so has mean $E$ and variance $E$. When $E$ is large, a Poisson count is approximately normal so $(O-E)/\sqrt{E}\sim N(0,1)$, hence this squared and summed over all possible class + target combinations gives a chi-squard distribution. 
$$
\chi^2 = \sum \frac{(O-E)^2}{E}
$$

5. **Cramer's-V**: If $n$ is large, even a trivially small association can return small p-value (association exists), we use Cramer's-V to quantify strength. Let $r,c$ be row and column counts, 
$$
V= \sqrt{\frac{\chi^2}{n\cdot (\min (r,c)-1)}}
$$
- This is on a 0-1 scale so we can compare. 

5. **Univariate ROC-AUC or MSE**:
- Use each feature on its own and look at ROC-AUC / PR-AUC for classifier. Or fit regression model and look at MSE. Misses complex relatinships.  

### Wrapper Methods 
Evaluates an ML model on various subsets of features to find the best one. They are greedy algortithms. 
- Can be very computationallly expensive
- Uses the ML model itself to source the feature set $\rightarrow$ better predictive performance. 
- Need an arbitrarily defined stopping criteria as to when to stop the search for the best subset. Threshold usually related to performance of ML model, or a predefined limit on the number of features. 

We have **sequential** or **randomised** searches. 

1. **Forward & Backward Selection**:
Either filter features by starting with all and reducing (backward) or starting from one feature and finding the best performing model, and then searching for the second best feature combined with the first one (forward).
    - Common metrics used as stopping criteria is the ROC-AUC for classification and the R-Squared for linear regression. 
    - This is a greedy algorithm as it evaluates all possible single, double, triple etc. feature combinations

2. **Exhaustive Feature Selection**: evaluates all possible feature combinations. Rarely used to computational cost. 

3. **Recursive Feature Selection**: Select features by recursively selecting a smaller and smaller set of features using feature importance metrics. Technically a hybrid wrapper and embedded method. 

### Embedded Methods: 
Learning algorithm takes advantage of its own variable selection process and performs feature selection and classification/prediction at the same time. Examples include Lasso and various types of tree-based algorithms.
    - Less computationally expensive as it is part of the model building prorcess
    - Can consider feature interactions 
    - Does not generalise to other models.

1. **Regularisation**: adds penalty to cost function which penalises number of features, provides automatic feature selection/importance.  
2. **Random Forest Importance**: 


4. **Feature Shuffling**
