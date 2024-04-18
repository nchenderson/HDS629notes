# Missing Data and Multiple Imputation {#missing-data}

* The book "Flexible Imputation of Missing Data" is a resource you also might find useful. It is available online at: https://stefvanbuuren.name/fimd/

## Missing Data in R and "Direct Approaches" for Handling Missing Data

* In many real-world datasets, it is very common to have missing values.

* In **R**, missing values are stored as `NA`, meaning **"Not Available"**.

---

* As an example, let's look at the `airquality` dataframe available in base **R**

```r
data(airquality)
head(airquality)
```

```
##   Ozone Solar.R Wind Temp Month Day
## 1    41     190  7.4   67     5   1
## 2    36     118  8.0   72     5   2
## 3    12     149 12.6   74     5   3
## 4    18     313 11.5   62     5   4
## 5    NA      NA 14.3   56     5   5
## 6    28      NA 14.9   66     5   6
```

* You can see that the **5th observation** for the `Ozone` variable is missing,
and the **5th and 6th observations** for the `Solar.R` variable are missing.

* You can use `is.na` to find which data entries have missing values.
    + If `is.na` returns `TRUE`, the entry is missing.
    + If `is.na` returns `FALSE`, the entry is not missing
    

```r
head( is.na(airquality) )
```

```
##      Ozone Solar.R  Wind  Temp Month   Day
## [1,] FALSE   FALSE FALSE FALSE FALSE FALSE
## [2,] FALSE   FALSE FALSE FALSE FALSE FALSE
## [3,] FALSE   FALSE FALSE FALSE FALSE FALSE
## [4,] FALSE   FALSE FALSE FALSE FALSE FALSE
## [5,]  TRUE    TRUE FALSE FALSE FALSE FALSE
## [6,] FALSE    TRUE FALSE FALSE FALSE FALSE
```

* Computing the sum of `is.na(airquality)` tells us how many missing values there are in this dataset.

```r
sum( is.na(airquality) )  ## 44 missing entries in total
```

```
## [1] 44
```

```r
dim( airquality )  ## Dataset has a total of 153 x 6 = 918 entries
```

```
## [1] 153   6
```

* Doing `apply(is.na(airquality), 2, sum)` gives us the number of missing values for each variable.

```r
apply( is.na(airquality), 2, sum)
```

```
##   Ozone Solar.R    Wind    Temp   Month     Day 
##      37       7       0       0       0       0
```

### Complete Case Analysis (Listwise Deletion)

* Suppose we wanted to run a regression with `Ozone` as the response and `Solar.R`,
`Wind`, and `Temp` as the covariates.

* If we do this in **R**, we will get the following result:

```r
air.lm1 <- lm( Ozone ~ Solar.R + Wind + Temp, data = airquality )
summary( air.lm1 )
```

```
## 
## Call:
## lm(formula = Ozone ~ Solar.R + Wind + Temp, data = airquality)
## 
## Residuals:
##     Min      1Q  Median      3Q     Max 
## -40.485 -14.219  -3.551  10.097  95.619 
## 
## Coefficients:
##              Estimate Std. Error t value Pr(>|t|)    
## (Intercept) -64.34208   23.05472  -2.791  0.00623 ** 
## Solar.R       0.05982    0.02319   2.580  0.01124 *  
## Wind         -3.33359    0.65441  -5.094 1.52e-06 ***
## Temp          1.65209    0.25353   6.516 2.42e-09 ***
## ---
## Signif. codes:  0 '***' 0.001 '**' 0.01 '*' 0.05 '.' 0.1 ' ' 1
## 
## Residual standard error: 21.18 on 107 degrees of freedom
##   (42 observations deleted due to missingness)
## Multiple R-squared:  0.6059,	Adjusted R-squared:  0.5948 
## F-statistic: 54.83 on 3 and 107 DF,  p-value: < 2.2e-16
```

---

* How is **R** handling all the **missing values** in `airquality`?

* As a default, `lm` performs a **complete case analysis** (sometimes referred to as listwise deletion).
     + (This is the default unless you changed the default `na.action` setting with `options(...)`)

* A **complete case analysis** for regression will **delete a rows** from the dataset if 
**any** of the variables used from that row have a missing value.
    + In this example, a row will be dropped if either the value of `Ozone`, `Solar.R`, `Wind`, or `Temp`
    is missing. 

* After these observations have been deleted, the usual regression parameters are estimated from the remaining **"complete" dataset**. 

---

* To remove **all rows** where there is at least one missing value, you can use
`na.omit`:

```r
complete.air <- na.omit( airquality )
## Check that there are no missing values
sum( is.na(complete.air) )  # This should be zero
```

```
## [1] 0
```

<!--* Because our regression model only uses the variables `Ozone`, `Solar.R`, `Wind`, and `Temp`,
we -->


---

* Let's now fit a linear regression using the complete dataset `complete.air`

```r
air.lm2 <- lm( Ozone ~ Solar.R + Wind + Temp, data = complete.air )
```

* Because **R** does a complete-case analysis as default, the estimated regression coefficients **should be the same** when using the "incomplete dataset" `airquality` as when using the "complete dataset" `complete.air`

```r
## The estimated regression coefficients should be the same
round(air.lm1$coefficients, 3)
```

```
## (Intercept)     Solar.R        Wind        Temp 
##     -64.342       0.060      -3.334       1.652
```

```r
round(air.lm2$coefficients, 3)
```

```
## (Intercept)     Solar.R        Wind        Temp 
##     -64.342       0.060      -3.334       1.652
```

### Other "Direct" Methods

* In general, doing a **complete-case analysis** is **not advisable**.
    + A complete-case analysis should really only be used if you are confident that the data are **missing completely at random (MCAR)**.
      
    + Roughly speaking, **MCAR** means that the probability of having a missing value is not related to
    **missing or observed** values of the data.
    
    + Section 2.5 gives a more formal definition of MCAR.

---

* A few other direct ways of handling missing data that you may have used or seen used in practice include:
    1. **Mean imputation**. Missing values are replaced by the value of the **mean** of that variable.
    
    2. **Regression imputation**. Missing values are replaced by a regression prediction from 
    the values of the other variables.

* Unless the data are **missing completely at random** (MCAR), each of these methods will produce biased estimates of the parameters of interest and generate incorrect standard errors.


---


## Multiple Imputation

### Short Overview of Multiple Imputation

* To summarize, **multiple imputation** consists of the following steps:
     1. Create $K$ different **"complete datasets"** where each dataset contains **no missing data**.
     
     2. For **each** of these $K$ complete datasets, compute the estimates of interest. 
     
     3. **"Pool"** these separate estimates to get final **estimates and standard errors**. 

---

* The nice thing about multiple imputation is that you can always
use your **original analysis approach**. 

* That is, you can just apply your original analysis method to each of the $K$ **complete datasets**.

* The complicated part in multiple imputation is generating the $K$ complete datasets in a **"valid"** or 
**"statistically principled"** way.

* Fortunately, there are a number of **R** packages that implement different approaches for creating
the $K$ **imputed datasets**.

### Multiple imputation with mice

* I will primarily focus on the **mice** package.
    + **mice** stands for **"Multivariate Imputation by Chained Equations"**

* The `mice` function within the **mice** package is the primary function
for performing **multiple imputation**.

* To use `mice`, just use `mice(df)` where `df` is the **name** of the dataframe.
    + (Set `print = FALSE` if you don't want it to print out the number of the iteration).
    + Choose a value of `seed` so that the results are reproducible.
    + Note that the **default number** of complete datasets returned is 5. This can be changed with the `m` argument. A value of `m` set to 5 or 10 is a typically recommendation for something that works well in practice. 
    
* Let's try running the `mice` function with the `airquality` dataset.


```r
library(mice)
imputed.airs <- mice(airquality, print=FALSE, seed=101)
```

---

* The object returned by `mice` will have a component called `imp` which is a list.
    
* Each component of `imp` is a **dataframe** corresponding to a single variable in the original dataframe 
     + This dataframe will contain the **imputed values** for the **missing values** of that variable.      

* For example, `imputed.airs$imp` will be a **list** with each component of the list being
one of the variables from `airquality`

```r
names( imputed.airs$imp )
```

```
## [1] "Ozone"   "Solar.R" "Wind"    "Temp"    "Month"   "Day"
```

---

* The `Ozone` component of `imputed.airs$imp` will be a dataframe with **37 rows** and **5 columns**.
    + This is because the `Ozone` variable had **37 missing values**, and there are **5 multiple imputations** (the default number in **mice**) 


```r
dim(imputed.airs$imp$Ozone)
```

```
## [1] 37  5
```

```r
## Imputed missing ozone values across the five multiple imputations
head(imputed.airs$imp$Ozone)
```

```
##     1  2  3  4  5
## 5   6  8 18  6 37
## 10 12 18 27 18 30
## 25  8 14  6 18 18
## 26 13  1 13 37 13
## 27 19 18  4 18 34
## 32 40 47 45 23 18
```

* The **row names** in `imputed.airs$imp$Ozone` correspond to the index of the observation
in the original `airquality` dataframe.

* For example, the 5th observation of the `Ozone` variable has 6 in the 1st imputation, 8 in
the 2nd imputation, 18 in the 3rd imputation, etc. ....

---

* Similarly, the `Solar.R` component of `imputed.airs$imp` will be a data frame 
   + This is because the `Solar.R` variable had **7 missing values**, and there are **5 multiple imputations**.


```r
dim(imputed.airs$imp$Solar.R)
```

```
## [1] 7 5
```

```r
## Imputed missing ozone values across the five multiple imputations
head(imputed.airs$imp$Solar.R)
```

```
##      1   2   3   4   5
## 5  131 285 274  92 139
## 6  127 248 175 167 175
## 11  71  71 238 115 284
## 27 238   8  49 223 238
## 96 258 203 229 223 291
## 97 313 259 274  83 272
```

* For example, the 5th observation of the `Solar.R` variable has 131 in the 1st imputation, 285 in
the 2nd imputation, 274 in the 3rd imputation, etc. ....

#### with(), pool(), complete()

* You could use the components of `imputed.airs$imp` to directly fit **5 separate regression** on the multiply imputed datasets and then average the results.

* However, this is much easier if you just use the **with** function from **mice**

```r
air.multi.imputelm <- with(imputed.airs, lm( Ozone ~ Solar.R + Wind + Temp))
```

* This will produce **5 different sets** of estimates of the regression coefficients:

```r
summary(air.multi.imputelm)
```

```
## # A tibble: 20 x 6
##    term        estimate std.error statistic  p.value  nobs
##    <chr>          <dbl>     <dbl>     <dbl>    <dbl> <int>
##  1 (Intercept) -90.3      19.1        -4.72 5.44e- 6   153
##  2 Solar.R       0.0642    0.0199      3.22 1.56e- 3   153
##  3 Wind         -2.31      0.550      -4.20 4.61e- 5   153
##  4 Temp          1.83      0.212       8.64 8.13e-15   153
##  5 (Intercept) -49.4      19.0        -2.60 1.02e- 2   153
##  6 Solar.R       0.0548    0.0196      2.79 5.92e- 3   153
##  7 Wind         -3.43      0.545      -6.28 3.44e- 9   153
##  8 Temp          1.46      0.211       6.92 1.27e-10   153
##  9 (Intercept) -56.7      19.0        -2.98 3.36e- 3   153
## 10 Solar.R       0.0646    0.0199      3.25 1.42e- 3   153
## 11 Wind         -3.22      0.546      -5.90 2.38e- 8   153
## 12 Temp          1.50      0.211       7.09 4.97e-11   153
## 13 (Intercept) -58.4      18.9        -3.09 2.41e- 3   153
## 14 Solar.R       0.0687    0.0199      3.46 7.08e- 4   153
## 15 Wind         -3.27      0.544      -6.01 1.35e- 8   153
## 16 Temp          1.57      0.210       7.50 5.33e-12   153
## 17 (Intercept) -58.9      22.2        -2.66 8.69e- 3   153
## 18 Solar.R       0.0404    0.0231      1.75 8.27e- 2   153
## 19 Wind         -3.31      0.636      -5.21 6.11e- 7   153
## 20 Temp          1.64      0.245       6.68 4.37e-10   153
```

* To get the "pooled estimates and standard errors" from these 5 different sets of regression coefficients
use the **pool** function from **mice**:

```r
summary( pool(air.multi.imputelm) )
```

```
##          term     estimate   std.error statistic       df      p.value
## 1 (Intercept) -62.73141323 26.25695768 -2.389135 16.63162 2.903387e-02
## 2     Solar.R   0.05855766  0.02400718  2.439173 36.55628 1.969933e-02
## 3        Wind  -3.10764673  0.75290758 -4.127527 16.76039 7.227036e-04
## 4        Temp   1.60026617  0.27155700  5.892929 23.89788 4.511087e-06
```

* The pooled **"final estimates"** of the regression coefficients are just the **means** of the estimated regression
coefficients from the 5 multiply imputed datasets.

* The pooled **standard error** for the $j^{th}$ regression coefficient is given by
\begin{equation}
(\textrm{pooled } SE_{j})^{2} = \frac{1}{K}\sum_{k=1}^{K} SE_{jk}^{2} + \frac{K+1}{K(K-1)}\sum_{k=1}^{K}(\hat{\beta}_{jk} - \bar{\hat{\beta}}_{j.})^{2}, 
\end{equation}
where $SE_{jk}$ is the standard error for the $j^{th}$ **regression coefficient** from the $k^{th}$ **complete dataset**,
and $\hat{\beta}_{jk}$ is the estimate of $\beta_{j}$ from the $k^{th}$ complete dataset.

---

* It is sometimes useful to actually **extract** each of the completed datasets.
    + This is true, for example, in **longitudinal data** where you may want to go back and forth between **"wide" and "long" formats**.
    
* To extract each of the completed datasets, you can use the `complete` function from **mice**.

* The following code will return the **5 complete datasets** from `imputed.airs`

```r
completed.airs <- mice::complete(imputed.airs, action="long")
```

* `action = "long"` means that it will return the 5 complete datasets as one dataframe with 
the individual datasets "**stacked** on top of each other."

* `completed.airs` will be a dataframe that has **5 times** as many rows as the `airquality` data frame
    + The variable `.imp` is an indicator of which of the 5 imputations that row corresponds to.

```r
head(completed.airs)
```

```
##   .imp .id Ozone Solar.R Wind Temp Month Day
## 1    1   1    41     190  7.4   67     5   1
## 2    1   2    36     118  8.0   72     5   2
## 3    1   3    12     149 12.6   74     5   3
## 4    1   4    18     313 11.5   62     5   4
## 5    1   5     6     131 14.3   56     5   5
## 6    1   6    28     127 14.9   66     5   6
```

```r
dim(completed.airs)
```

```
## [1] 765   8
```

```r
dim(airquality)
```

```
## [1] 153   6
```

---

* Using `complete.airs`, we can compute the multiple imputation-based estimates of the regression coefficients **"by hand"**
   + This should give us the same results as when using `with`


```r
BetaMat <- matrix(NA, nrow=5, ncol=4)
for(k in 1:5) {
    ## Find beta.hat from kth imputed dataset
    BetaMat[k,] <- lm(Ozone ~ Solar.R + Wind + Temp, 
                      data=completed.airs[completed.airs$.imp==k,])$coefficients
}
round(colMeans(BetaMat), 3)  # compare with the results from using the "with" function
```

```
## [1] -62.731   0.059  -3.108   1.600
```



## What is MICE doing?

* Suppose we have data from $q$ variables $Z_{i1}, \ldots, Z_{iq}$.

* Let $\mathbf{Z}_{mis}$ denote the **entire collection** of missing observations and $\mathbf{Z}_{obs}$
the **entire collection** of observed values, and let $\mathbf{Z} = (\mathbf{Z}_{obs}, \mathbf{Z}_{mis})$.

* The **basic idea behind** multiple imputation is to, in some way, generate samples $\mathbf{Z}_{mis}^{(1)}, \ldots, \mathbf{Z}_{mis}^{(K)}$ from a flexible probability model $p(\mathbf{Z}_{mis}|\mathbf{Z}_{obs})$
    + $p(\mathbf{Z}_{mis}|\mathbf{Z}_{obs})$ represents the conditional distribution of $\mathbf{Z}_{mis}$ given the observed $\mathbf{Z}_{obs}$.

---

* A valid estimate $\hat{\theta}$ of the parameter of interest $\theta$ can often be thought of as
a posterior mean:
    + $\hat{\theta} = E(\theta|\mathbf{Z}_{obs})$ is the expectation of $\theta$ given the observed data $\mathbf{Z}_{obs}$.

* Then, $\hat{\theta}$ can be expressed as:
\begin{eqnarray}
\hat{\theta} &=& E( \theta |\mathbf{Z}_{obs}  )
= \int E\Big\{ \theta \Big| \mathbf{Z}_{obs}, \mathbf{Z}_{mis} \Big\} p(\mathbf{Z}_{mis}|\mathbf{Z}_{obs}) d\mathbf{Z}_{mis} \nonumber \\ 
&\approx& \frac{1}{K} \sum_{k=1}^{K} E\Big\{ \theta \Big| \mathbf{Z}_{obs}, \mathbf{Z}_{mis}^{(k)} \Big\} 
\end{eqnarray}

---

* There are two main approaches for setting up a model for the **conditional distribution** of $\mathbf{Z}_{mis}|\mathbf{Z}_{obs}$.

* One approach is to directly specify a **full joint model** for $\mathbf{Z} = (\mathbf{Z}_{mis}, \mathbf{Z}_{obs})$
     + For example, you could assume that $\mathbf{Z}$ follows a multivariate a normal distribution.
     
     + This approach is not so straightforward when you have variables of **mixed type**: some continuous,some
     binary, some categorical, etc. ...

---

* Let the vector $\mathbf{Z}_{j} = (Z_{1j}, \ldots, Z_{nj})$ denote all the "data" (whether observed or unobserved)
from variable $j$.

* The **fully conditional specification** (FCS) approach specifies the distribution of each variable $\mathbf{Z}_{j}$ conditional on the remaining variables $\mathbf{Z}_{-j}$.
    + The FCS approach is the one used by **mice**.

* With the FCS approach, we assume models for $q$ different conditional distributions 
\begin{eqnarray}
p(\mathbf{Z}_{1}&|&\mathbf{Z}_{-1}, \boldsymbol{\eta}_{1}) \nonumber \\
p(\mathbf{Z}_{2}&|&\mathbf{Z}_{-2}, \boldsymbol{\eta}_{2}) \nonumber \\
&\vdots& \nonumber \\
p(\mathbf{Z}_{q}&|&\mathbf{Z}_{-q}, \boldsymbol{\eta}_{q})
\end{eqnarray}

---

* With **mice**, the parameters $\eta_{j}$ and the missing values for each variable $\mathbf{Z}_{j,mis}$ are updated **one-at-a-time** via a kind of **Gibbs sampler**.

* All of the missing values can be imputed in **one cycle** of the Gibbs sampler.

* Multiple cycles are repeated to get **multiple completed datasets**.

---

* The default model for a continuous variable $\mathbf{Z}_{j}$ is to use **predictive mean matching**. 

* The default model for a binary variable $\mathbf{Z}_{j}$ is **logistic regression**.

* Look at the `defaultMethod` argument of `mice` and @buuren2010 for more
details about how to change these default models.

## Longitudinal Data

* A direct way to do multiple imputation with **longitudinal data** is to use mice 
on the dataset stored in **wide format**.

* Remember that in **wide format**, each row corresponds to a **different individual**.

* Applying multiple imputation to the wide-format dataset can account for the fact that
observations across individuals will be **correlated**.

---

* Let's look at the **ohio** data from the **geepack** package again

```r
library(geepack)
data(ohio)
head(ohio)
```

```
##   resp id age smoke
## 1    0  0  -2     0
## 2    0  0  -1     0
## 3    0  0   0     0
## 4    0  0   1     0
## 5    0  1  -2     0
## 6    0  1  -1     0
```

---

* The **ohio** dataset is in **long format**. We need to first convert this into **wide format**. 

* With the **tidyr** package, you can convert from **long to wide** using **spread**:
    + (Use **gather** to go from **wide to long**)

```r
library( tidyr )
ohio.wide <- spread(ohio, key=age, value=resp)

## Change variable names to so that ages go from 7 to 10
names(ohio.wide) <- c("id", "smoke", "age7", "age8", "age9", "age10")
head(ohio.wide)
```

```
##   id smoke age7 age8 age9 age10
## 1  0     0    0    0    0     0
## 2  1     0    0    0    0     0
## 3  2     0    0    0    0     0
## 4  3     0    0    0    0     0
## 5  4     0    0    0    0     0
## 6  5     0    0    0    0     0
```

* The variable `age7` now represents the value of `resp` at age 7, 
`age8` represents the value of `resp` at age 8, etc...


---

* **reshape** from base **R** can also be used to go from **long to wide**

```r
# Example of using reshape
#ohio.wide2 <- reshape(ohio, v.names="resp", idvar="id", timevar="age", direction="wide")
```

---

* The **ohio** dataset does not have any missing values. 

* Let's introduce missing values for the variable `resp` values
by assuming that the **probability** of being missing is positively
related to smoking status.


* Let $R_{ij}$ be an **indicator of missingness** of `resp` for individual $i$
at the $j^{th}$ follow-up time.

* When randomly generating missing values, we will assume that: 
\begin{equation}
P( R_{ij} = 1| \textrm{smoke}_{i}) 
= \begin{cases} 
0.05 & \textrm{ if } \textrm{smoke}_{i} = 0 \\
0.3 & \textrm{ if } \textrm{smoke}_{i} = 1
\end{cases}
(\#eq:missingdat-ohio)
\end{equation}

---



* To generate missing values according to assumption \@ref(eq:missingdat-ohio), we can use the following R code:
    + We will call the new data frame `ohio.wide.miss`

```r
ohio.wide.miss <- ohio.wide
m <- nrow(ohio.wide.miss) ## number of individuals in study
for(k in 1:m) {
    resp.values <- ohio.wide[k, 3:6]  # values of resp for individual k
    if(ohio.wide[k,2] == 1) {  # if smoke = 1
        Rij <- sample(0:1, size=4, replace=TRUE, prob=c(0.7, 0.3))
    } else { # if smoke = 0
        Rij <- sample(0:1, size=4, replace=TRUE, prob=c(0.95, 0.05))
    }
    resp.values[Rij==1] <- NA # insert NA values where Rij = 1
    ohio.wide.miss[k, 3:6] <- resp.values
}
```




```r
head(ohio.wide.miss, 10)
```

```
##    id smoke age7 age8 age9 age10
## 1   0     0    0    0   NA     0
## 2   1     0    0    0    0     0
## 3   2     0    0    0    0     0
## 4   3     0    0    0    0     0
## 5   4     0    0    0    0     0
## 6   5     0    0    0    0     0
## 7   6     0    0    0    0     0
## 8   7     0    0   NA    0     0
## 9   8     0    0    0    0     0
## 10  9     0    0    0    0     0
```

* `ohio.wide.miss` now has 257 missing entries

```r
sum( is.na(ohio.wide.miss))
```

```
## [1] 296
```

---

* Before using **multiple imputation** with `ohio.wide.miss`, let's look at the regression coefficient
estimates that would be obtained with a **complete case analysis**.

* To use `glmer` on the **missing-data version** of `ohio`, we need to first convert `ohio.wide.miss` back into **long form**:

```r
ohio.miss <- gather(ohio.wide.miss, age, resp, age7:age10)
ohio.miss$age[ohio.miss$age == "age7"] <- -2
ohio.miss$age[ohio.miss$age == "age8"] <- -1
ohio.miss$age[ohio.miss$age == "age9"] <- 0
ohio.miss$age[ohio.miss$age == "age10"] <- 1
ohio.miss <- ohio.miss[order(ohio.miss$id),]  ## sort everything according to id
ohio.miss$age <- as.numeric(ohio.miss$age)
head(ohio.miss)
```

```
##      id smoke age resp
## 1     0     0  -2    0
## 538   0     0  -1    0
## 1075  0     0   0   NA
## 1612  0     0   1    0
## 2     1     0  -2    0
## 539   1     0  -1    0
```

* Let's use a **random intercept** model as we did in our earlier discussion of generalized linear mixed models:

```r
## Complete case analysis

library(lme4)
ohio.cca <- glmer(resp ~ age + smoke + (1 | id), data = ohio.miss, family = binomial)

# Now look at estimated regression coefficients for complete case analysis:
round(coef(summary(ohio.cca)), 4)
```

```
##             Estimate Std. Error z value Pr(>|z|)
## (Intercept)  -3.8009     0.4346 -8.7459   0.0000
## age          -0.1580     0.0783 -2.0165   0.0437
## smoke         0.2333     0.3402  0.6857   0.4929
```

---

* Now, let's use **mice** to create 10 **"completed versions"** of `ohio.wide.miss`

```r
imputed.ohio <- mice(ohio.wide.miss, m=10, print=FALSE, seed=101)
```

* For the case of **longitudinal data**, we probably want to actually extract each
complete dataset.
   + (This is because many of the analysis methods such as `lmer` assume the data is in long form).

* This can be done with the following code:

```r
completed.ohio <- mice::complete(imputed.ohio, "long")
head(completed.ohio)
```

```
##   .imp .id id smoke age7 age8 age9 age10
## 1    1   1  0     0    0    0    0     0
## 2    1   2  1     0    0    0    0     0
## 3    1   3  2     0    0    0    0     0
## 4    1   4  3     0    0    0    0     0
## 5    1   5  4     0    0    0    0     0
## 6    1   6  5     0    0    0    0     0
```

* `completed.ohio` will be a **dataframe** that has **10 times** as many rows as the original `ohio.wide` data frame

```r
dim(ohio.wide)
```

```
## [1] 537   6
```

```r
dim(completed.ohio)
```

```
## [1] 5370    8
```

* The variable `.imp` in `completed.ohio` is an indicator of which of the 10 "imputed datasets" this is from:

```r
table( completed.ohio$.imp ) # Tabulate impute indicators
```

```
## 
##   1   2   3   4   5   6   7   8   9  10 
## 537 537 537 537 537 537 537 537 537 537
```

---

* For **each** of the 10 complete datasets, we need to **convert** the wide dataset 
into long form before using `glmer`:


```r
## Multiple imputation-based estimates of regression coefficients 
## for the missing version of the ohio data.
BetaMat <- matrix(NA, nrow=10, ncol=3)
for(k in 1:10) {
    tmp.ohio <- completed.ohio[completed.ohio$.imp==k,-c(1,2)]
    
    tmp.ohio.long <- gather(tmp.ohio, age, resp, age7:age10)
    tmp.ohio.long$age[tmp.ohio.long$age == "age7"] <- -2
    tmp.ohio.long$age[tmp.ohio.long$age == "age8"] <- -1
    tmp.ohio.long$age[tmp.ohio.long$age == "age9"] <- 0
    tmp.ohio.long$age[tmp.ohio.long$age == "age10"] <- 1
    tmp.ohio.long$age <- as.numeric(tmp.ohio.long$age)
    
    ohio.tmpfit <- glmer(resp ~ age + smoke + (1 | id), data = tmp.ohio.long, 
                         family = binomial)
    BetaMat[k,] <- coef(summary(ohio.tmpfit))[,1]
}
```

* The **multiple imputation-based** estimates of the regression coefficients for 
the missing version of **ohio** are:

```r
round(colMeans(BetaMat), 4)
```

```
## [1] -3.5711 -0.1066  0.2865
```

* Compare the above regression coefficients with those from the **complete-case** analysis.

## Different Missing Data Mechanisms

* For this section, we will consider the setup where we have **$n$ "observations" and $q$ "variables"**:
denoted by $Z_{i1}, \ldots, Z_{iq}$, for $i = 1, \ldots, n$.

* Let $\mathbf{Z}_{mis}$ denote the collection of missing observations and $\mathbf{Z}_{obs}$
the collection of observed values, and let $\mathbf{Z} = (\mathbf{Z}_{obs}, \mathbf{Z}_{mis})$.

* The variables $R_{ij}$ are defined as
\begin{equation}
R_{ij} = 
\begin{cases}
1 & \textrm{ if } Z_{ij} \textrm{ is missing } \\
0 & \textrm{ if } Z_{ij} \textrm{ is observed } 
\end{cases}
\end{equation}

### Missing Completely at Random (MCAR)

* The **missingness mechanism** is said to be MCAR if
\begin{equation}
P(R_{ij} = 1|\mathbf{Z}_{obs}, \mathbf{Z}_{mis}) = P(R_{ij}=1)
\end{equation}

### Missing at Random (MAR)

* The **missingness mechanism** is said to be MAR if:
\begin{equation}
P(R_{ij} = 1|\mathbf{Z}_{obs}, \mathbf{Z}_{mis}) = P(R_{ij}=1|\mathbf{Z}_{obs})
\end{equation}

* If missingness is follows either MAR or MCAR, direct use of multiple imputation
is a valid approach.


### Missing not at Random (MNAR)

* If the **missingness mechanism** is classified as **missing not at random** (MNAR), the probability 
$P(R_{ij} = 1|\mathbf{Z}_{obs}, \mathbf{Z}_{mis})$ cannot be factorized into a simpler form.

* If the missingness is MNAR, direct use of multiple imputation may be invalid and 
modeling of the missingness mechanism using subject-matter knowledge may be needed.

* Use of multiple imputation with a sensitivity analysis is one approach to consider.

---



