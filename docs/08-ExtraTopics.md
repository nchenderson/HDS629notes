# Variable Importance Measures {#extra-topics}

---


## Partial Dependence Plots

- Beyond variable importance, assessing the impact of a particular covariate can be tricky with
more "black box" prediction method.

-  For these situations, [**partial dependence plots**]{style="color:#D96716"} are a useful way tool. 

- Suppose we have a method that takes covariate vector $\mathbf{x}$ as input and generates a prediction $\hat{f}(\mathbf{x})$.
     - For example, $\hat{f}(\mathbf{x})$ could be the output of random forest or from a boosting method

-  For this method, the [**partial dependence function**]{style="color:#D96716"} for variable $k$ at point $u$ is defined as 
\begin{equation}
    \rho_{k}( u ) = \frac{1}{n}\sum_{i=1}^{n}\hat{f}(\mathbf{x}_{i,-k}, u)
\end{equation}

-   $\mathbf{x}_{i,-k}$ - all of the covariates for person $i$ [**except**]{style="color:#D96716"} for covariate $k$.

-   $(\mathbf{x}_{i,-k}, u)$ - setting $x_{ik} = u$ and using the observed values for all other covariates.

---

- Plotting $\rho_{k}(u)$ versus $u$ captures how changing $x_{ik}$ influences the estimated prediction.

- Note that partial dependence plots are mostly useful for **continuous covariates**. 



## Uncertainty in Variable Importance Measures

* Many machine learning/penalized regression methods generate
measures of variable importance.
    + Random forests generate variable importance scores.
    
    + Partial dependence plots are useful for assessing the impact of a covariate for any learning method.
    
    + Lasso has selection/magnitude of regression coefficients.
    
* These variable importance measures do not usually come
with a measure of uncertainty for each variable importance score.

* For example, you may want to report a confidence interval 
for the VIMP scores.

### Subsampling for Random Forest VIMP scores

* A general approach to assessing the **uncertainty** of a variable importance measure
is to use some form of repeated **subsampling/sample splitting**.

* The basic idea is to draw subsamples, and for each subsample
compute variable importance scores. Then, estimate the variance
of each variable importance score using their variation across different subsamples
    + See @ishwaran2019 for more details of this approach. 

---

* Steps in subsampling approach:
    1. Draw a subsample of **size b** from the **original** dataset. Call it $D_{s}$.
    
    2. Using random forest on dataset $D_{s}$, compute VIMP scores $I_{s,j}$ for variables $j=1,\ldots,p$.
    
    3. Repeat steps 1-2 $S$ times. This will produce $I_{s,j}$ for all subsamples $s = 1, \ldots, S$ and all variables $j = 1, \ldots, p$.
    
    4. Estimate the **variance** of $I_{s,j}$ with the quantity
    \begin{equation}
       \hat{v}_{j} = \frac{b}{nK} \sum_{s=1}^{S}\Big( I_{s,j} - \bar{I}_{.,j}  \Big)^{2}
    \end{equation}

* A $95\%$ **confidence interval** for the variable importance of variable $j$ will then be
\begin{equation}
I_{j} \pm 1.96 \times \sqrt{\hat{v}_{j}}
\end{equation}
   + Here, $I_{j}$ is the variable importance score from the full dataset.

---

* To test this out, we will use the **diabetes** data. 
     + This can be obtained from https://hastie.su.domains/CASI/data.html
     


* This dataset has 442 **observations** and 10 **covariates**

* The outcome variable of interest is **prog**


```r
dim(diabetes)
```

```
## [1] 442  11
```

```r
head(diabetes)
```

```
##   age sex  bmi map  tc   ldl hdl tch      ltg glu prog
## 1  59   1 32.1 101 157  93.2  38   4 2.110590  87  151
## 2  48   0 21.6  87 183 103.2  70   3 1.690196  69   75
## 3  72   1 30.5  93 156  93.6  41   4 2.029384  85  141
## 4  24   0 25.3  84 198 131.4  40   5 2.123852  89  206
## 5  50   0 23.0 101 192 125.4  52   4 1.863323  80  135
## 6  23   0 22.6  89 139  64.8  61   2 1.819544  68   97
```

---

* Let's first fit a **randomForest** to the entire dataset and plot
the variable importance measures

```r
library(randomForest)
```

```
## randomForest 4.7-1.1
```

```
## Type rfNews() to see new features/changes/bug fixes.
```

```r
rf.full <- randomForest(prog ~ ., data=diabetes)
varImpPlot(rf.full)
```

<img src="08-ExtraTopics_files/figure-html/unnamed-chunk-3-1.png" width="528" />

* You can extract the actual values of the variable importance scores by
using the `importance` function.

```r
Imp.Full <- importance(rf.full)
Imp.Full ## This is a 10 x 1 matrix
```

```
##     IncNodePurity
## age     142733.87
## sex      31747.77
## bmi     579528.45
## map     299960.63
## tc      145300.41
## ldl     154247.81
## hdl     207463.96
## tch     163943.49
## ltg     558491.96
## glu     210089.27
```


---

* Now, let's compute variable importance scores across $S = 100$ subsamples (each of size 50)
and store it in a $10 \times S$ matrix called `Imp.Subs`

```r
S <- 5
b <- 100
Imp.Subs <- matrix(0, nrow=nrow(Imp.Full), ncol=S)
rownames(Imp.Subs) <- rownames(Imp.Full)
for(k in 1:S) {
    ## sample without replacement
    subs <- sample(1:nrow(diabetes), size=b)
    diabetes.sub <- diabetes[subs,]
    rf.sub <- randomForest(prog ~ ., data=diabetes.sub)
    Imp.Subs[,k] <- importance(rf.sub)
}
```

* From `Imp.Subs`, we can compute the **variance estimates** $\hat{v}_{j}$.

```r
imp.mean <- rowMeans(Imp.Subs)
vhat <- (b/nrow(diabetes))*rowMeans((Imp.Subs - imp.mean)^2)  
print(vhat)
```

```
##         age         sex         bmi         map          tc         ldl 
##   3802632.6    438003.3  59610723.7 232968479.9   3260099.6   1317631.1 
##         hdl         tch         ltg         glu 
##  26016161.4  16557560.8 129036456.5   4840373.7
```

---

* We can now report **confidence intervals** for the variable importance scores:

```r
vi.upper <- Imp.Full[,1] + 1.96*sqrt(vhat)
vi.lower <- Imp.Full[,1] - 1.96*sqrt(vhat)
VIMP_CI <- cbind(Imp.Full[,1], vi.lower, vi.upper)
colnames(VIMP_CI) <- c("estimate", "lower", "upper")
VIMP_CI[order(-VIMP_CI[,1]),]
```

```
##      estimate     lower     upper
## bmi 579528.45 564395.69 594661.22
## ltg 558491.96 536227.49 580756.42
## map 299960.63 270044.55 329876.70
## glu 210089.27 205777.10 214401.44
## hdl 207463.96 197466.77 217461.14
## tch 163943.49 155968.05 171918.92
## ldl 154247.81 151997.96 156497.66
## tc  145300.41 141761.49 148839.34
## age 142733.87 138911.80 146555.94
## sex  31747.77  30450.61  33044.94
```

### Stability Selection for Penalized Regression

* Let's use the **lasso** with penalty $\lambda = 10$ on the `diabetes` data:

```r
library(glmnet)
```

```
## Loading required package: Matrix
```

```
## Loaded glmnet 4.1-8
```

```r
diabet.mod <- glmnet(x=diabetes.sub[,1:10],y=diabetes.sub$prog, 
                     lambda=20)
```

* We can look at the estimated coefficients to see which variables were **selected**

```r
coef(diabet.mod)
```

```
## 11 x 1 sparse Matrix of class "dgCMatrix"
##                     s0
## (Intercept) -97.960901
## age           .       
## sex           .       
## bmi           3.197026
## map           .       
## tc            .       
## ldl           .       
## hdl           .       
## tch           3.278505
## ltg          75.102716
## glu           .
```

* The **selected** variables are those with nonzero coefficients:

```r
# Look at selected coefficients ignoring the intercept:
selected <- abs(coef(diabet.mod)[-1]) > 0
selected
```

```
##  [1] FALSE FALSE  TRUE FALSE FALSE FALSE FALSE  TRUE  TRUE FALSE
```

---

* When thinking about how certain you might be about a given set of selected variables, one
natural question is how would this set of selected variables change if
you re-ran the lasso on a different subset.

* You might have more confidence in a variable that is consistently selected
across random subsets of your data.

---

* For a given choice of $\lambda$, the **stability** of variable $j$ is 
defined as
\begin{equation}
\hat{\pi}_{j}(\lambda) = \frac{1}{S}\sum_{s=1}^{S} I( A_{j,s}(\lambda) = 1),
\end{equation}
where ...

* $A_{j,s}(\lambda) = 1$ if variable $j$ in data subsample $s$ is **selected** and

* $A_{j,s}(\lambda) = 0$ if variable $j$ in data subsample $s$ is **not selected**

* @meinshausen2010 recommend drawing subsamples of size $n/2$.

---

* The quantity $\hat{\pi}_{j}(\lambda)$ can be thought of as an 
estimate of the **probability** that variable $j$ is in the "selected set" of variables. 

* Variables with a large value of $\hat{\pi}_{j}(\lambda)$ have a greater **"selection stability"**.

* You can plot $\hat{\pi}_{j}(\lambda)$ across different values of $\lambda$ to get 
a sense of the range of selection stability. 

---

* For the `diabetes` data, let's first compute an $S \times m \times 10$ array,
where the $(k, h, j)$ element of this array equals $1$ if variable $j$ 
was selected in subsample $k$ with penalty term $\lambda_{h}$:

```r
nsamps <- 200
b <- floor(nrow(diabetes)/2)
nlambda <- 40 ## 40 different lambda values
lambda.seq <- seq(0.1, 20.1, length.out=nlambda)
## Create an nsamps x nlambda x 10 array
SelectionArr <- array(0, dim=c(nsamps, nlambda, 10))
for(k in 1:nsamps) {
   subs <- sample(1:nrow(diabetes), size=b)
   diabetes.sub <- diabetes[subs,]
   for(h in 1:nlambda) {
      sub.fit <- glmnet(x=diabetes.sub[,1:10],y=diabetes.sub$prog, 
                        lambda=lambda.seq[h])
      selected <- abs(coef(sub.fit)[-1]) > 0
      SelectionArr[k,h,] <- selected
   }
}
```

* From this **array**, we can compute a matrix containing
selection probability estimates $\hat{\pi}_{j}(\lambda)$.
   + The $(j, h)$ component of this matrix has the value $\hat{\pi}_{j}(\lambda_{h})$


```r
SelectionProb <- matrix(0, nrow=10, ncol=nlambda)
rownames(SelectionProb) <- names(diabetes)[1:10]
for(h in 1:nlambda) {
   SelectionProb[,h] <- colMeans(SelectionArr[,h,]) 
}
```

* The first few columns of `SelectionProb` look like the following:

```r
SelectionProb[,1:5]
```

```
##      [,1]  [,2]  [,3]  [,4]  [,5]
## age 0.965 0.775 0.640 0.495 0.365
## sex 1.000 1.000 1.000 1.000 0.990
## bmi 1.000 1.000 1.000 1.000 1.000
## map 1.000 1.000 1.000 1.000 1.000
## tc  0.965 0.870 0.790 0.665 0.570
## ldl 0.925 0.270 0.265 0.285 0.290
## hdl 0.880 0.925 0.970 0.985 0.990
## tch 0.950 0.630 0.430 0.295 0.195
## ltg 1.000 1.000 1.000 1.000 1.000
## glu 1.000 0.910 0.850 0.805 0.735
```

---

* We can now plot the stability measures as a function of $\lambda$

```r
## Convert to long form:
df <- data.frame(varname=rep(names(diabetes)[1:10], each=nlambda),
                 selection.prob=c(t(SelectionProb)), lambda=rep(lambda.seq, 10))
head(df)
```

```
##   varname selection.prob    lambda
## 1     age          0.965 0.1000000
## 2     age          0.775 0.6128205
## 3     age          0.640 1.1256410
## 4     age          0.495 1.6384615
## 5     age          0.365 2.1512821
## 6     age          0.245 2.6641026
```


```r
library(ggplot2)
```

```
## 
## Attaching package: 'ggplot2'
```

```
## The following object is masked from 'package:randomForest':
## 
##     margin
```

```r
ggplot(df) + aes(x=lambda, y=selection.prob, 
                 group=varname, color=varname) + geom_line()
```

<img src="08-ExtraTopics_files/figure-html/unnamed-chunk-15-1.png" width="480" />

