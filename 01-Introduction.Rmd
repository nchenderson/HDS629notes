# Mixed Models for Longitudinal Data Analysis {#mixed-models}


---
  
## Methods for Analyzing Longitudinal Data {#sec:methods-overview}

* **Longitudinal data** refers to data that:
    + Has multiple individuals/subjects.  
    
    + Each individual has multiple observations that were taken across time. 

* Typically the outcomes of interest are denoted as $Y_{ij}$.
    + $Y_{ij}$ - outcome for individual $i$ at time $t_{j}$.
    
    + The $i^{th}$ individual has $n_{i}$ observations: $Y_{i1}, \ldots, Y_{in_{i}}$.

---

* Most of the well-known methods for analyzing longitudinal
data can be classified into one of the three following
categories: 
    + **Random effects/mixed models**, 
    + **Marginal models**, 
    + **Transition models**

* **Random effects/Mixed Models**
    + "Random effects" are added to the regression model describing
    the outcomes for each individual.
    
* **Marginal models**
    + Only mean of $Y_{ij}$ and correlation structure of $(Y_{i1}, \ldots, Y_{in_{i}})$ is
    modeled. Generalized estimating equations (GEEs) are used for estimating model parameters.
    

* **Transition models**


## Mixed Models for Continuous Outcomes

* If each $Y_{ij}$ is a **continuous outcome** and we were to 
build a regression model without any random effects, we might assume something like:
\begin{equation}
Y_{ij} = \beta_{0} + \mathbf{x}_{ij}^{T}\boldsymbol{\beta} + e_{ij}
\end{equation}

* $\mathbf{x}_{ij} = (x_{i1}, \ldots, x_{ip_{i}})$ is the vector
of covariates for individual $i$ at time $j$.

---

* The regression model () assumes that the same 
mean function holds for all individuals in the study.

* This often ignores  


## Advantages of using random effects

* BLUPs

* Automatically accounts for within-subject correlation 

* Flexibility of regression plus "regularization" 

## Generalized linear mixed models (GLMMs)

* Generalized linear models (GLMs) are used to handle "non-continuous" data
that can't be reasonably modeled with a Gaussian distribution.

* The most common cases encountered in practice where GLMs are needed: **binary** outcomes
and **count** outcomes.

* For **binary** outcomes, responses are assumed to follow a Binomial distribution
and the log-odds for this Binomial distribution is modeled with a linear regression.

* For **count** outcomes, responses are assumed to follow a Poisson or a negative binomial distribution
and the log of the mean is modeled with a linear regression.

---

* In a GLMM with binary outcomes, one 

* $Y_{ij}|u_{i} \sim \textrm{Binomial}( p_{ij} )$.

\begin{equation}
\log\Big( \frac{ p_{ij} }{1 - p_{ij} } \Big) = \beta_{0} + \mathbf{x}_{ij}^{T}\boldsymbol{\beta} + \mathbf{z}_{i}^{T}\mathbf{u}_{i}
\end{equation}

* Count data $Y_{ij}|u_{i} \sim \textrm{Poisson}(\mu_{ij})$
\begin{equation}
\log( \mu_{ij} ) = \beta_{0} + \mathbf{x}_{ij}^{T}\boldsymbol{\beta} + \mathbf{z}_{i}^{T}\mathbf{u}_{i}
\end{equation}

## Fitting Mixed Models and GLMMs in **R**

* The **lme4** package is probably the most general package
for fitting mixed models and GLMMs.

* aa






    
