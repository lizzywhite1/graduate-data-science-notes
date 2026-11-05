# Resampling Methods 
Resampling methods involve repeatedly drawing samples from the training data and refitting the model of interest. 

## Cross-Validation 
Want to estimate the test error rate of a specific statistical learning method given a set of observations. We need the test set to be suffciently large in order to get a reliable test error rate. We cannot use the train error rate as this will be underestimate of the true error rate. 

Cross validation can be used either to get a reliable estimate of the true test error, or to find the minimum of the test error to compare models or parameters. 

### Validation Set Approach
Divides the avaliable set of data into a training set and a validation (or hold-out) set. The model is fit on the training set and the fitted model is evaluated on the validation set. The validation set provides an estimate for the test error rate, and can be used to tune the model parameters, or for model selection. 

**Drawbacks**
- The validation test error rate can be highly variable
- Validation test error may overestimate real test error, as if were were to train model on entire dataset, we would likely get better results. 

### Leave One Out Cross Validation (LOOCV)
Trains the model on the entire dataset, minus one observation, $(x_1,y_1)$, records validation test error. Repeats for all of the $n$ data points. 
$$
CV_{(n)}=\frac 1 n \sum ^n _{i=1} MSE _i
$$
![](/graduate-data-science-notes/machine_learning/images/LOOCV.png)
*Introduction to Statistical Learning with Applications in Python (2023)* 

**Pros**
- Less bias, LOOCV tends to not overestimate the test error rate, as we are training on more data. 
- Less variability in results compared to validation set approach. 

**Drawbacks** 
- Can be expensive to implement since the model has to be fit $n$ times. 

![](/graduate-data-science-notes/machine_learning/images/LOOCV_curves.png)
*Introduction to Statistical Learning with Applications in Python (2023)*

### K-Fold Cross Validation 
Divides the set of observation into $k$ groups and use $k-1$ to train the model and the last one to evaluate the test error rate. Repeat over the $k$ groups. 
$$
CV_{(k)}= \frac 1 k \sum ^k _{i=1} MSE_i
$$
![](/graduate-data-science-notes/machine_learning/images/k-fold.png)
*Introduction to Statistical Learning with Applications in Python (2023)*

LOOCV is k-fold cross validation with $n$ folds. In practise we usually set $k=5$ or $k=10$. Use $k=10$ when you want more reliable, lower bias estimates and have adaquete compute. Use $k=5$ when the dataset is large, or model fitting is slow. 

**Bias Variance Tradeoff Cross Validation**
LOOCV produces approximately unbiased estimates of the test error rate, making it the lowest bias approach, with the validation set approach having the highest.

LOOCV has a higher variance than k-fold CV. LOOCV is in effect averaging the outputs of $n$ fitted models, each trained on almost identitical datasets. Hence, model outputs are highly positively correlated. As the mean of highly correlated results, has higher variance than the mean of results which are less correlated with one another, k-fold CV tends to have lower variance. 

## Bootstrap
Widely used technique to quantify the uncertainty associated with a given sample estimate. 

In an ideal world we could resample from the population as many times as we wanted, to obtain estimates for the data. In practise, we use bootstrapping. Instead of obtaining independent data sets from the population, bootstrapping samples uniformly with replacement from the $n$ observations in the sample $B$ times.   