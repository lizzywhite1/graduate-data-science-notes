# Classification 
Goal is to assign each observation to one of K labels given an outputed predicted probability (or log odds). We have two types of classification models 
1. Generative: models the joint distribution $p(x,y) = p(x|y)p(y)$ 
    - e.g. Gaussian Mixture Models 
    - Models the distribution of individual classes 
    - Pros: generate new data, handle missing data (marginalise over it), incorperate prior knowledge via prior distribution
    - Cons: computationally expensive, strong assumptions, usually worse at pure classification
2. Discriminative: models the conditional probability $p(y|x$ directly
    - More common, makes fewer assumptions about the distribution of the data
    - e.g. Logistic regression, SVMs, random forests, neural nets 
    - Pros: higher accuracy (directly minimises classification error), fewer assumptions, resource effcieint 
    - Cons: cannot generate data, struggles w missing data, needs more data to start otherwise it may overfit

## Metrics 
Accuracy alone isnn't a good metric, especially for imbalanced class labels. E.g. predicting a rare cancer, you could simply predict no cancer and have high accuracy in the dataset. 
We would want to incooperate information regarding the false negatives, ie. when we incorrectly predict someone DOES NOT have cancer but they do. A false positive is called a Type 1 erorr and a False Negative is called a Type 2 error. 

|                         | **Predicted: Churn (1)**             | **Predicted: No churn (0)**           |
|-------------------------|--------------------------------------|---------------------------------------|
| **Actual: Churn (1)**   | **TP** – True Positive               | **FN** – False Negative (Type II error) |
| **Actual: No churn (0)**| **FP** – False Positive (Type I error) | **TN** – True Negative              |

Example: 
- **FN (Type II)** = missed churner → lost revenue
- **FP (Type I)** = false alarm → wasted retention offer

To do with positive predictions:
- **Recall** = TP / (TP + FN): of actual churners, how many we caught. How many positives did we detect?
- **Precision** = TP / (TP + FP): of predicted churners, how many really churned. How many positives that were predicted were actually correct?

These two metrics are what we want to focus on.

To do with negative predictions:
- **Negative Predicted Value**: TN / (TN + FN): Of all negatives predicted, how many were actually correct. 
- **Specificity**: TN / (TN + FP): How many negatives did we successfull predict.

In the case where we want to maximise both equally, we want to look at F1-Score

$$F1 = 2* \frac{precision * recall}{precision + recall}$$

Which is 2x the harmonic mean of the two. Harmonic mean gives a larger weighting to smaller values. 

We can also evaluate using the ROC Curve and AUC score. The ROC curve plots the true positive rate versus the false positive rate. 
- TPR : TP / (TP + FN) aka the recall 
- FPR: FP / (FP + TN)

Alternatively, in the case that dataset is heavily imbalanced or the positive class is rare and the main focus, can use the Precision-Recall curve. 

## Models
### Baseline Models 

## Naive Bayes 

## Logistic Regression Models 

## 