# Statistical Experiments and Significance Testing
The designing of experiments is a fundamental practise of statistics. We want to **design an experiment which rejects or fails to reject a hypothesis**. 

## A/B testing 
An **A/B test** is an experiment with two groups to establish which of the two treatments, products or procedures is superior. Our typical hypothesis is that the treatment performs better than the control. 

Often one of the two treatments is the standard treatment (or no treatment), this is called the **control group**. We need a control group because we want to ensure that we hold all other things equal so we can be confident the observed effect of treatement is from the treatmennt alone. The other group is called **treatment group**. For each group "subjects" are assigned randomly to one of the treatments.

During A/B testing, we measure a **metric** such as conversion rate or uplift, and then we calculate the **test statistic** based on this metric to test our hypothesis.

## Hypothesis Tests & Statistical Significance 
A **hypothesis test** is used to assess whether sample data provide enough evidence against a hypothesis about the population. A hypothesis is a claim about the value of a parameter or a characteristic of a population.

A hypothesis test involves the following steps:

1. **State a null and an alternative hypothesis.** We will either reject the null in favour of the alternative, or fail to reject the null, meaning there is not sufficient evidence against it (not that it is true).
2. **Calculate a test statistic** whose distribution under the null hypothesis is known, and use it to calculate the corresponding p-value.
3. **Compare** the p-value to a chosen significance level, $\alpha$, and reject the null if $p \le \alpha$.

These tests can be either one-tailed,
$$
H_0: \mu = \mu_0 \quad H_1: \mu > \mu_0 \quad \text{or} \quad H_1: \mu < \mu_0,
$$
or two-tailed,
$$
H_0: \mu = \mu_0 \quad H_1: \mu \ne \mu_0.
$$
We can also have **composite** hypotheses, e.g. $H_0: \mu \le \mu_0 \quad H_1: \mu > \mu_0$, or the other way around.

We can simply say if the test statistic, $x\in R$ where $R$ is the rejection (or critical) region we reject the null.

A **Type I** error occurs when we reject the null hypothesis but it was actually true, a false positive. The Type I error rate is the probability of incorrectly rejecting $H_0$.

### P-values and confidence intervals 
A p-value is the probability of the observed results under the null hpothesis. If the p-value is small, then it means the probability of that event occuring under the null is small, as we would likely reject the null.

Under the null hypothesis, p-values should be approximately uniformly distributed. Under the alternative hypothesis, the histogram of p-values is skewed towards $0$. 

For a test of size $\alpha$, the confidence interval is such that the probability of the estimate lies in the given interval is $1-\alpha$. 

## Errors, Power and Sample Size
The **size** of the test, $\alpha$, is the probability of rejecting $H_0$, given it is true. We choose $\alpha$ to be small, say $\alpha = 0.05$, which caps the Type I error rate at or below $\alpha$. The more serious a type I error, the smaller $\alpha$ should be. 

The **power** of the test is the probability of rejecting $H_0$ given $H_1$ is true. It measures the test's ability to detect a true difference or effect. The power function is 
$$
\beta(\theta) =\mathbb P(x\in R|\theta)
$$
so $1-\beta$ is the probability of a type II error if $\beta = \beta(\theta_1)$. 

## Test Statistics 
### One Sample Mean
Suppose we have a sample mean for a given sample, and we want to conduct a hypothesis test regarding the mean of the population. 

A two-tailed test would be,
$$
H_0:\mu =\mu_0\\ H_1: \mu\ne \mu_0
$$
A one-tailed test could be 
$$
H_0:\mu = \mu_0\\ H_1: \mu> \mu_0
$$
or 
$$
H_0:\mu = \mu_0\\ H_1: \mu< \mu_0
$$

When we know the true population variance, $\sigma$, the z test statistic is, 
$$
z=\frac{\bar x -\mu_0}{\sigma /\sqrt n }.
$$
For an upper one tailed test our rejection region is if $z\ge z_\alpha$ , and for lower one tailed test its $x\le -z_\alpha$. For two-tailed test its $z\le - z_{\alpha/2}$ or $z\ge z_{\alpha / 2}$. 

If we do not know $\sigma$ but the sample size is large (e.g. $n\ge 30$), we can estimate $\sigma$ and use the sample standard deviation. If the sample size is not large, we use a t-test.

This result hinges on the central limit theorem, i.e. for a large enough sample size the sample mean is approximately normal. The Z test statistic is distributed according to a standard normal. For large samples, a t-test would yield similar results. 

### One Sample Proportion
If we have count data, and we assume each observation is independent, we can assume $X= \sum ^n  _{i=1} Y_i \sim \text{Binomial}(n,p)$. We can show our estimate of the proportion of successes $\hat p = X/n$ is unbiased, and that it has an approximately normal distribution with standard deviation 
$$
\sigma_{\hat p} = \sqrt{p(1-p)/n} 
$$ 
When $n$ is large, the test statistic is 
$$
Z= \frac{\hat p -p_0}{\sqrt {p_0(1-p_0)/n}}
$$
where $p_0$ is what we assume under null. 

As $\hat p$ is simply the mean of $n$ independent Bernoulli trials, we know $\hat p$ is approximately normal by CLT (assuming $n$ is large). We require $np_0$ and $np_0(1-p_0)$ are both greater than $10$. If sample size is small use exact binomial test.

### Two-Sample Means 
Now we are looking at the difference between means, $\Delta= \bar x -\bar y $. If we know the population variances and the population is normal OR the sample size is large enough, we can use the two sample z-statistic, 
$$
H_0: \Delta _0=\mu_2-\mu_1\\
Z=\frac{\bar x -\bar y -\Delta_0}{\sqrt{\frac{\sigma_1^2}{m}+\frac{\sigma_2^2}{n}}}.
$$
Otherwise, if we do not know population variances we can use the sample standard deviations and use a t-test.

### Difference in Proportions 
Calculate our two proportions $\hat p_1,\hat p_1$. Our null hypothesis, $H_0$, is that $p_2-p_1=0$. The groups are independent so $\text{var}(p_2-p_1)=\frac{p_1(1-p_1)}{n_1}+\frac{p_2(1-p_2)}{n_2}$. Additionally, under $H_0$, we estimate $\hat p=\frac{x_1+x_2}{n_1+n_2}$. Under $H_0$, the standard deviation is hence 
$$
SE_0=\sqrt{\hat p(1-\hat p)\left (\frac{1}{n_1}+\frac{1}{n_2}\right )}.
$$

This gives the test statistic, 
$$
Z= \frac{\hat p_2-\hat p_1}{SE_0}\approx N(0,1)\quad \text{under }H_0
$$

If the count's are small use Fisher's exact test. 

### Equality of Variances
If $X_1,X_2$ are independent chi-squared random variables with degrees of freedom $\nu_1, \nu_2$, then 
$$
F = \frac{X_1/\nu_1}{X_2/\nu_2}\sim F_{\nu_1,\nu_2}
$$
follows an $F$ distribution. 

As $(n-1)S^2$ is the sum of squared deviations from the mean (which are standard normally distributed), and we know sums of squared standard normal distributions are chi-squared distributed then 
$$
\frac{s_1^2/\sigma_1^2}{s_2^2/\sigma_2^2}\sim F_{n_1-1, n_2-1}
$$

Under $H_0:\sigma_1^2=\sigma_2^2$, we have $F= \frac{s_1^2}{s_2^2}$. 

### ANOVA 


### Chi-Squared Goodness of Fit and Indepedence Test
**Goodness of Fit**

$$
\chi ^ 2=\sum \frac{(O-E)^2}{E}
$$
$O$ is the observed count, $E$ is the expected count under the null, usually $n\times p$. Degrees of freedom is $k-1$. 


### Shapiro wilks
### Mann-Kendall tests 

## Multiple Testing