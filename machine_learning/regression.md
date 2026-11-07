### Decision Trees
Decision trees are inteperable and simple algorithms 
which split the predictor space into regions, wherein the mean of the training target values is the prediction for an unlabelled observations. The learning algorithm aims to find the best splits of the predictor space which minimises the loss.

**Terminology**:
- The final regions are known as the **terminal nodes** or **leaves** of the tree. 
- The points where the split is made along the tree are known as the **internal nodes**. 
- Segments of the tree that connect each node are called **branches**. 

![](/graduate-data-science-notes/machine_learning/images/DT_example.png)
*Introduction to Statistical Learning with Applications in Python (2023).*

**Regression with Decision Trees**
- Prediction is taken as the mean of the target value within that node /region.

**Tree Fitting for Regression**

The goal is to find regions of the predictor space $R_1,...,R_j$ which minimise the loss function, in this case residual sum of squares (RSS)
$$
\sum _{j=1}^J\sum _{i\in R_j}(y_i-\hat y_{R_j})^2
$$
where $\hat y_{R_j}$ is the mean of the responses in $R_j$.


It is computationally, infeasible to consider every possible partition, so we usually take a top down / greedy approach, known as **recursive binary splitting**. 
1. Find the first split $\{X | X_j <> s\}$ and $\{X|X_j \ge s\}$ which results in the greatest reduction in the RSS. 
2. Then for each of the two splits, find the next best split for each which minimises the RSS. 
3. Repeat until stopping criterion reached.

**Tree Pruning**

DTs are likely to overfit especially if the resulting tree is too complex. Some solutions
1. Reduce the complexity of the tree by limiting the tree depth 
2. Ensure that each split reduces RSS by some considerable amount. This is short-sighted as one worthless split may lead to important split further on.

Better solution is tree pruning. Grow a very large tree, $T_0$, and then prune to contain a subtree. 
- Find a subtree which results in best performance during cross validation or on the validation set.

However, looking at every subtree is too computationally demanding. Hence, we introduce **cost complexity pruning** (also known as **weakest link pruning**). Once we have built $T_0$, we find the subtree $T\subset T_0$ which for a given $\alpha$ minimises, 
$$
\sum ^{|T|} _{m=1}\sum _{i: x_i\in R_m}(y_i-\hat y_{R_m})+\alpha |T|
$$
- The $\alpha |T|$ term is the penalty we induce for the number of terminal nodes $|T|$. As $\alpha$ increases there is a cost to have too many terminal nodes, and so the size of $T$ will shrink. 
- When $\alpha=0$ the subtree will equal $T_0$ as this quantity is just the training error (which $T_0$ was built to minimise). 

**Algorithm**
1. Use RBS to grow a large DT, $T_0$, on training data. Stop learning when stopping criterion reached. 
2. Apply cost complexity pruning to $T_0$ to obtain a sequence of best subtrees as a function of the tuning/penalty parameter, $\alpha$. 
3. Use K-Fold cross validation to choose best $\alpha$.  For $k=1,...,K$, 
    1. Repeat steps 1-2 on k-th fold training data. 
    2. Evaluate MSE on the data left out of the k-th fold as a function of $\alpha$ 
    3. Average the results for each value of $\alpha$, and pick the $\alpha$ which minimises MSE.
4. Return the subtree from Step 2 that corresponds to chosen $\alpha$. 

![](/graduate-data-science-notes/machine_learning/images/CV_trees.png)*Introduction to Statistical Learning with Applications in Python (2023).*