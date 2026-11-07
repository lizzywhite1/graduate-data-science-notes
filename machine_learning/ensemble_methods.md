# Bagging, Random Forests, Boosting, and Bayesian Additive Regression Trees
An **ensemble** method is an approach which combines many simple models (known as **weak learners**) to obtain a simple and potentially powerful model. Can be applied with both regression and classification trees. Types of ensemble methods. 

## Bagging
Bootstrap aggregation or bagging is a procedure for **reducing the variance** of a statistical learning method. This makes them useful for decision trees, which suffer from high variance.

Averaging a set of observations reduces the variance. Given a set of $Z_1,...,Z_n$ independent, with variance $\sigma^2$ , the variance of the mean, $\bar Z$, is $\sigma^2/n$. If we obtain $B$ seperate bootstrapped training sets, build seperate models on each, and then average the predictions, $\hat f^1 (x),...,\hat f^B (x)$, gives us a single lower variance model, 
$$
\hat f _{avg}(x)=\frac 1 B \sum ^B _{b=1} \hat f ^b(x).
$$
This is called bagging.

This allows us to build several deep trees, which are not pruned, whilst having a lower variance model. 

One can show, that each weak learner uses approximately $\frac 2 3$ of the available data, so we can record the out of bag (OOB) error rate, for the $i$ th observation using trees not fit on this observation. This yields around $\frac B 3 $ predictions for each observation. 

## Variable Importance Measures
Bagging reduces the inteperability of the decision trees. One can use the RSS, for regression trees, or Gini Index (or entropy) for classification trees to get an overall summary of model fit. 

For bagging regression trees, record the total reduction in RSS due to splits over a given predictor, averaged over the $B$ trees. Large values indicate an important predictor. Similarly, for bagging classification trees, look at the total reduction in the Gini index or entropy, averaged over the $B$ trees. 

## Random Forests
Random forests (RFs) provide an improvement over bagged trees. Bagged trees consider all possible features, which means trees can be highly correlated (resulting in a higher model variance), especially if their is a **dominant predictor**, as every tree will likely use this feature. Averaging highly correlated predictions will not result in a huge reduction in variance. 

Instead, random forests use bagged trees, but only consider a random sample of $m$ predictors for each bootstrapped samples. Typically $m\approx \sqrt p$. 

**Choice of $m$**
- Small values of $m$ are useful when considering a large number of correlated features.
- Too small values of $m$ may result in individual trees, with poor predictions, and prediction will vary massively. 

## Boosting
Instead of growing trees based on different bootstrapped samples of the data, boosting grows trees sequentially, fitting trees based on information from the previous one. Each tree is grown using a modified version of the original dataset.

We fit each decision tree on the residuals of the previous decision tree output. Each tree can be fairly small and contain only a few number of terminal nodes. By fitting small trees to the residuals, we slowly improve $\hat f$ in areas where it does not perform well. 

**Algorithm**
1. Set $\hat f(x) = 0$ and $r_i = y_i$ for all $i$ in training set. 
2. For $b=1,...,B$ :

    a. Fit a tree $\hat f ^b$ with $d$ splits ($d+1$ terminal nodes) to $(X,r)$ 
    
    b. Update $\hat f$ by adding a shrunken version of $\hat f ^b$ 
    $$
    \hat f (x) \leftarrow \hat f(x)+ \lambda \hat f^b (x)
    $$ 

    c. Update the residuals
    $$
    r_i\leftarrow r_i -\lambda \hat f ^b (x_i)
    $$
3. Output the boosted model, 
$$
\hat f (x)=\sum ^B _{b=1} \lambda \hat f ^b (x)
$$

**Tuning Parameters**
1. Number of trees $B$. Boosting can overfit if $B$ is too large. Random forests benefit from $B$ being large. Use cross-validation to select $B$ 
2. Shrinkage parameter, $\lambda >0$, typically between $0.001$ or $0.01$. The smaller the $\lambda$, the more $B$ trees are required to learn.
3. The number, $d$, of splits within the tree, which controls the complexity of the boosted ensemble. $d=1$ can work well, wherin, each tree is just a stump. 

## Bayesian Additive Regression Tree



