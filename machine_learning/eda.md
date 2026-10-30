## Exploratory Data Analysis 
1. Generate questions about data
2. Search for answers by visualising, summarising, transforming or modelling 
3. Refine your questions and generate new questions 

Before you plot, some useful things to do,

```python
print(df.shape)

df.info(verbose = True, show_counts = True)

df.describe(include = 'all')
```

Following statistics can be used to measure **location** 
- Mean: good for normally distributed data, or data with no outliers/heavy tails. Can be weighted. Trimmed mean, to find the mean after dropping certain extreme values.
- Median: 50th percentile of the ordered data. Better measure of location when data has heavy tails/outliers. Can be weighted. 

We may down weight a variable if we know the reading is less accurate then the others. We may upweight a particular reading if it represents an underrepresented part of the group. 

To measure **variability**:
1. Deviations: Absolute difference between observation and mean.
2. Variance: is the sum of the squared deviations from the mean / (n-1). 
3. Standard Deviation : square root of variance. In the same units as the data.
4. Mean/Median absolute deviation 
5. Range: difference between smallest and largest value 
6. Percentile: P-th percentile is the percent of values that take on this value or less. (1-P)th percentile is the percent of values that take on this value or more. 
7. IQR: Difference between 75th percentile and 25th percentile. 

N-1 is the degrees of freedom, it is required to make the standard deviation an unbiased estimate. 

We can use the groupby method,

```python
df.groupby('group')['value'].agg(['mean', 'median', 'std']).round(2).T
df.groupby('key').aggregate({'feature1': 'min', 'feature2': 'max'})
df.groupby('target')['groups'].value_counts(normalize = True).T

# Can filter observations out 
def filter_out(x): 
    # return only groups where the std of feature is greater than 1 
    return x['feature'].std > 1 

df.groupby('key').filter(x)

# To transform data to subtract group mean/median from mean
df.groupby('group').transform(lambda x: x - x.mean)
df.groupby('group')['col'].transform(lambda x: x - x.mean)

# .apply() lets you apply arbitrary functions to results 
def func(df):
    df['secondary_revenue'] /=   df['total_revenue'].sum()
    return df 

df.groupby('key').apply(func)

# or equivalently 
df["secondary_revenue"] / df.groupby("key")["total_revenue"].transform("sum")
```

When working with categorical data we can use 
- Mode; used for categorical data, which one appears the most? Use .value_counts() to see most frequently occuring.
- Expected Value.

Use Pearson's correlation coeffcient to assess the strength of the linear relationship between features.

**Types of plots** 
- Histograms: use to see distribution of numeric variables, can create different bin widths, can colour by category. Usually count/frequency of how many fall into one bin. Density plots are smoothed versions of histogram. Usually a kernel density estimat eis used. 
- Barplot: for counts of categorical data, it is the frequency or proportion of occurnace. Can break down into additional sub-categories. 
- Scatter plot: to see relationship between two numeric features, can colour by catgeory
- Boxplots: to see distribution statistics of numeric features, a violin plot allows you to see the shape of the distribution 
- Tile/Heatmap: can compare the value of a numeric feature for two different categorical features, also can be used to visualise correlation
- Contingency tables to get counts for two different categroical features 
- Contour Plots & Hexaganol binning: when too many data points, scatterplots get too dense. 

Correlation formula:
$$
\rho _{XY} = \frac{\sum (x_i-\bar x)(y_i-\bar y)}{(N-1)s_Xs_Y}=\frac{s_{XY}}{s_{X}s_{Y}}
$$
where $s_X, s_Y$ are the sample standard deviations, and s_{XY} is sample correlation.


Useful functions to plot