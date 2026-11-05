# Classification 
The goal of classification is to assign one of $K$ labels a given predicted probability (or log odds). We have two types of classification models, 
1. **Generative**: models the **joint** distribution $P(x,y) = P(x|y)P(y)$. 
    - *e.g. Gaussian mixture models*
    - **Pros**: as we model the distribution of individual classes we can generate new data, we can handle missing data by just marginalising over that feature, incorperate prior knowledge via our choice of prior distribution
    - **Cons**: can be computationally expensive, has strong assumptions on prior, usually worse at pure classification tasks.
2. **Discriminative**: models the conditional probability $P(y|x)$ directly.
    - *E.g. Logistic regression, SVMs, random forests, neural nets*
    - **Pros**: higher accuracy (directly minimises classification error), fewer assumptions (more flexible), resource effcient. 
    - **Cons**: cannot generate data, struggles ithw missing data, needs more data to start otherwise it may overfit. 

## Metrics 
Accuracy alone isnn't a good metric, especially for imbalanced class labels. For example when predicting a rare cancer, you could simply predict no cancer and have high accuracy in the dataset. 

We would want to incooperate information regarding the false negatives, ie. when we incorrectly predict someone DOES NOT have cancer but they do. The two types of error a binary classification model can encounter is a false positive is called a Type I erorr and a false negative is called a Type II error. Below is a confusion matrix, where one would input the number in each, 

|                         | **Predicted: Churn (1)**             | **Predicted: No churn (0)**           |
|-------------------------|--------------------------------------|---------------------------------------|
| **Actual: Churn (1)**   | **TP** – True Positive               | **FN** – False Negative (Type II error) |
| **Actual: No churn (0)**| **FP** – False Positive (Type I error) | **TN** – True Negative              |

Example: 
- **FN (Type II)** = missed churner → lost revenue
- **FP (Type I)** = false alarm → wasted retention offer

Metrics related to the true positives:
- **Recall (Sensitivity)** = $\frac{TP}{(TP + FN)}$: how well does the model recall the positive cases out of all positives. Also known as the True Positive Rate (TPR)
- **Precision** = $\frac{TP}{(TP + FP)}$: how many of the model's positive predictions are correct.

These are the two key metrics we usually focus on. We may aim to maximise both equally, which we can use the F1- Score for:
$$F1 = 2* \frac{precision * recall}{precision + recall}$$
Which is the harmonic mean of the two. The harmonic mean gives a larger weighting to smaller values. 

Additional metrics:
- **Negative Predicted Value**: $\frac{TN}{TN + FN}$: Of all negatives predicted, how many were actually correct? 
- **Specificity (True Negative Rate)**: $\frac{TN}{(TN + FP}$: How many negatives did we successfully predict.
- **False Positive Rate**: $\frac{FP}{FP + TN}$ (or $1-\text{Specificity}$), equivalent to Type I error rate (how many false alerts are we having?)

### ROC & Precision-Recall Curves 
Using a $50\%$ classification threshold may not always be the best choice, especially in situations where the cost of a false positive or false negative is disproportionately worse. There is always a trade-off between precision and recall when deciding on the threshold, increasing the classification threshold and making a positive classification less likely may lower the FPs but could increase the FNs (missing positive cases), and vice versa. 
![](/graduate-data-science-notes/machine_learning/images/error_rate_thresh.png)*Introduction to Statistical Learning with Applications in Python (2023)*

The ROC curve is a common way to visualise the two types of errors given each threshold for the classifier. It plots the TPR (recall/specificity) against the FPR (1-specificity). 

![](/graduate-data-science-notes/machine_learning/images/roc_curve.png)*Introduction to Statistical Learning with Applications in Python (2023)*

We often measure the area under the curve (AUC). Higher the AUC the better, we aim for our classifier to hug the top corner of the graph, as this is where we minimise FPR and maximise TPR. 
The dotted 45 degree diaganol line represents where TPR = FPR (and AUC = 0.5). This line represents where a classifier is no better than random guessing. If 50% of the time we correctly predict positive class our of all positives, but we also have a false alert rate of 50% then our predicts would be not better than just flipping a fair coin, and choosing positive or negative. Models with an ROC curve close or on this line, provide no useful insight for the modelling problem at hand. 

Alternatively, in the case that dataset is heavily imbalanced or the positive class is rare and the main focus, one can use the precision-recall curve. As an example, consider 10,000 emails where 1% (100) are spam. If our model correctly detects 80 (recall is 80%) but also flags 400 legitimate emails (FPR = 400/9,900 = 4%), our FPR looks great but the precision of the model is 80/(480) = 16% so 84% of the models alerts are false alarms, which our ROC curve hides. 

Use PR curves when you have a large class imbalance, false alarms are costly, or when the negative class doesn't matter that much. 

### Uplift charts 


## Models
### Baseline Models 
We can use dummy classifiers to set a baseline. Gives us a trivial baseline. 

Examples include
1. Always predict the most frequent. DummyClassifier('most_frequent')
2. Predict the class according to a multinomial distribution with same support as training data, DummyClassifier('stratified'). 

```python 
# Example of fitting a DummyClassifier given X, y train. 
```

## Generative Models 
### Naive Bayes 
Naive Bayes is an extremely fast, interperable and simple classifcation algorithm. It is based on Bayes theorem, and the assumption of conditional independence of the predictors given the class label. Recall, Bayes Theorem
$$
P(Y|\textbf X) = \frac{P(\textbf X|Y) P(Y)}{P(\textbf X)}.
$$ 
where $\textbf X = (X_1,...,X_k)$. With the conditional independence assumption 
$$
P(Y|\textbf X) = \frac{P(Y) \prod ^k _{i=1} P( X_i|Y)}{P(\textbf X)}\propto P(Y) \prod^k _{i=1} P(X_i|Y).
$$ We aim to find the argument which maximises this, 
$$
\hat y \propto \arg \max _y P(y) \prod^k _{i=1} P(x_i|y).
$$

The conditional independence assumption makes it a good algorithm when the size of the training data is small or the dataset is high dimensional. This is because for $k$ binary predictors we have $2^k$ possible combinations of predictors, and so our ML model needs $2^k$ data points to model these $2^k$ interactions. But if $P(X_1,...X_k|Y)= \prod _{i=1}^k P(X_i|Y)$ then we only need $k$ data points. 

In order to make a classification we could look at 
$$
\frac{P(Y=1|X)}{P(Y=0|X)}=\frac{P(X|Y=1)P(Y=1)}{P(X|Y=0)P(Y=0)}
$$ 
which we need to specify the distribution of $ X |Y$. We can estimate distribution of Y by looking at class proportions. In practise, this conditional independence assumption almost never holds, but in certain scenarios, e.g. text classification where the predictors are words, Naive Bayes can be effective. 

There are various types of [Naive Bayes models](https://scikit-learn.org/stable/modules/naive_bayes.html) depending on the problem at hand.

```python 
# Example of Gaussian Naive Bayes

```

**Considerations and Model Outputs**

## Discriminative Models
### Logistic Regression Models 
Used to model the probability of binary responses, and is one of the most frequenttly used parameteric classification methods. 

We assume that our dependent variable, $y_i$ is distributed according to a Bernoulli distribution with probability $\mu_i$, given we know $\textbf x_i$, our features. 
$$
y_i \mid \boldsymbol{x}_i \sim \mathrm{Bernoulli}(\mu_i), \\ \mu_i = \mathbb{P}(y_i = 1 \mid \boldsymbol{x}_i) = \zeta'(\eta_i).
$$ 
where $\eta_i$ is the linear predictor $\textbf x_i^T\boldsymbol \beta$. 

The function $\zeta':\mathbb R\to (0,1)$ is the logistic function defined as 
$$\zeta '(t) = \frac{e^t}{1 + e^{t}},$$
which is the first derivative of $\zeta(t)=\log(1+e^t)$. This yields the logistic regression model in its explicit form,
$$\mu_i = \mathbb{P}(y_i=1 \mid \boldsymbol{x}_i) \\= \zeta'(\boldsymbol{x}_i^\top \boldsymbol{\beta}) \\= \frac{e^{\boldsymbol{x}_i^\top \boldsymbol{\beta}}}{1 + e^{\boldsymbol{x}_i^\top \boldsymbol{\beta}}}.$$

![](/graduate-data-science-notes/machine_learning/images/log_reg_curve.png)*Logistic regression passes a linear combination of features through sigmoid function which maps any real number to a probability between (0,1) and equivalently it models the log-odds (logit) as linear in the features.*

Alternatively, the model can also be parametrised using the logit link function, which expresses the log-odds as a linear combination of the covariates
$$    
\eta_i = \text{logit}(\mu_i) \\= \log\left(\frac{\mu_i}{1-\mu_i}\right) \\= \sum_{j=1}^p x_{ij} \beta_j.
$$

The loss function is the log-loss, which is equivalent to MLE, 
$$
L\left(\boldsymbol \beta \right) = \sum ^n _{i=1}y_i \log \left(\frac{1}{\mu_i}\right)+ (1-y_i)\log \left(\frac{1}{1-\mu_i}\right)
$$

With linear regression, we know $\beta_1$ represents the average change in the dependent variable with a one unit variable change in $X_1$. In the logistic regression model, a one unit change in $X_1$, results in change by $\beta_1$ units of the log odds, or it multiplies the odds by $e^{\beta_1}$. 

**Pros & Cons**
- **Pros**: like linear regression, logistic regression is highly interpretable; the predicted probability is easy to explain. Quick to compute. Great baseline models. 
- **Cons**: Simple model makes it a high bias low variance model, when the decision boundary is not linear. When features are highly correlated, $\boldsymbol \beta$ will not be accurate (use regularisation, removal of features etc. to address). Standard ML estimates not guaranteed to exist for high dimensional data. 

```python 
import statsmodels.api as sm
import numpy as np

from sklearn.linear_model import LogisticRegression 

# for model summary 
X_train_c = sm.add_constant(X_train)
lr_model = sm.Logit(y_train, X_train_c)
lr_result = lr_model.fit()

print(lr_result.summary()

# for cross validation results 

cv = StratifiedKFold(n_splits=10, shuffle=True, random_state=42)
 
lr = LogisticRegression(max_iter = 1000, C = np.inf)

lr_scores = cross_validate(lr, X_train_c, y_train)

```

**Modelling Considerations**
Just as with linear regression, the standard error of coefficients can explode due to **multicollinearity**. Multicollinearity occurs when two or more independent predictor variables are highly correlated. Multicollinearity is problematic due to the way in which the coefficients are estimated in logistic regression models, as they require the inversion of the weighted matrix $X^TWX$ which becomes ill-conditioned when two or more features are (close to being) collinear. This causes small changes in the data, to result in large changes to coefficients, and the standard error of the effected coefficients to be large. 

One way to detect collinearity between two variables is to use a correlation matrix. A more effective way to detect multicollinearity between two and more variables, is using the **Variance Inflation Factor (VIF)**.
$$
\text{VIF}(\hat{\beta}_j)=\frac{1}{1-R^2_{X_j|X_{-j}}}
$$
where $R^2_{X_j|X_{-j}}$ is the R-squared of the logistic regression model of $X_j$ regressed onto $X_{-j}$ remaining predictors. Recall,
$$
R^2 = 1- \frac{SS_{res}}{SS_{total}}= 1- \frac{\sum (y_i-\hat y_i)^2}{\sum (y_i-\bar y)^2}
$$
This tells us how much $X_j$ explains the remaining predictors. If this is $0$, meaning $X_{-j}$ have no explanatory power for $X_j$ then the VIF is $1$. As the R-squared increases towards $1$, the VIF $\to\infty$. *A rule of thumb is usually a VIF of >5, is moderate multicollinearity, and VIF of >10 is server multicollinearity*. 

Additionally, it is often worth scaling features. 

```python
# To get the VIF of features in Pandas DataFrame


```

**Interpreting Outputs** 

### Support Vector Machines

### Classification Trees 
Decision trees are interpreable and simple. They work by finding the best rules to split the predictor space to find the region which results in maximising the objective function. 

DTs and Tree based methods are best used when their is non-linear and complex relationships between features, if their is a linear relationship between features and response, then linear models may perform better. 

**Terminology**: 
- The final regions are known as the **terminal nodes** or **leaves** of the tree. 
-The points where the split is made along the tree are known as the **internal nodes**. 
- Segments of the tree that connect each node are called **branches**. 

Classification trees are very similar to regression trees, except this time we are trying to predict a categorical response. Our prediction for each reigion of the split predictor space, is the class of the most commonly occuring label in that region. We are usually also interested in class proportion in that region. 

Similarly we use Recursive Binary splitting as we did in the regression tree, except this time we use a different criterion for the splits instead of RSS.

**Loss Functions for Classification Trees**

An obvious criterion would be the classification error rate, as this is ultimately what we are trying to minimise in the final tree. 
$$
E= 1 - \max _k (\hat p_{mk})
$$
where $\hat p_{mk}$ represents the proportion of training observations in the $m$-th region that are from the $k$ th class. 
- However, it turns out this criterion is **not suffciently** sensitive for tree-building. 

Usually, two other loss functions are used in practise. 
1. The **Gini Index**,
$
G= \sum ^K _{k=1}\hat p_{mk}(1-\hat p _{mk})
$, is a measure of the total variance across the $K$ classes. If each $\hat p_{mk}$ is close to zero or one, then $G$ will be small, and hence the node is very "pure". 
2. **Entropy** is given by $D= - \sum ^K_{k=1} \hat p_{mk}\log \hat p_{mk}$. Since $0\le \hat p_{mk}\le 1$ then $0\le - \hat p_{mk} \log \hat p_{mk}$. One can see that if $\hat p_{mk}$ is close to zero or one for all $k\in 1,...,K$ then entropy will be close to zero. 

These two approaches are more sensitive to node purity then misclassifcation error rate. Any of these three metrics can be used to prune the tree, if prediction accuracy is the ultimate goal then use misclassification error rate to prune. Prune the trees the same as we did for regression trees. 

**Pros**:
- Interperable and some beleive they mirror human decision making 
- Can be displayed graphically 
- Can easily handle categorical variables without need to create dummy variables 

**Cons**:
- Lower predictive accuracy due to overfitting 
- Not robust 





