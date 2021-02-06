# Missing Data and Multiple Imputation {#missing-data}

## Missing Data in R and "Direct Approaches" for Handling Missing Data

* In a wide range of datasets, it is very common to encounter missing value.

* In **R**, missing values are stored as `NA`, meaning "Not Available".

---

* For example, look at the `airquality` dataframe available in base **R**

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

* How is **R** handling all the **missing values** in `airquality`.

* As a default, `lm` conducts a **complete case analysis** (sometimes referred to as listwise deletion).
     + (This is the default unless you changed the default `na.action` setting with `options(...)`)

* A **complete case analysis** for regression will first **delete all rows** from the dataset if 
any of the variables used from that row have a missing value.
    + In this example, a row will be dropped if either the value of `Ozone`, `Solar.R`, `Wind`, or `Temp`
    is missing. 

* After these observations have been deleted, the usual regression is fit to the remaining "complete" dataset. 

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

* Because our regression model only uses the variables 


---

* Let's now fit a linear regression using the complete dataset `complete.air`

```r
air.lm2 <- lm( Ozone ~ Solar.R + Wind + Temp, data = complete.air )
```

* The estimated regression coefficients **should be the same** when using 
the "incomplete dataset" `airquality` as when using the "complete dataset" `complete.air`

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
    + A complete-case analysis should really only be used if you are confident that the data are missing 
      completely at random (MCAR).

---

* A few other direct ways of handling missing data that you may have used or seen used in practice include:
    1. **Mean imputation**. Missing values are replaced by the value of the **mean** of that variable.
    
    2. **Regression imputation**. Missing values are replaced by a regression prediction from 
    the values of the other variables.


---


## Multiple Imputation

* To summarize, **multiple imputation** consists of the following steps:
     1. Create $K$ different "complete datasets" which contain no missing data.
     
     2. For each of these $K$ complete datasets, compute the estimates of interest. 
     
     3. "Pool" these separate estimates to get a final estimate. 

---

* The nice thing about multiple imputation is that you can always
use your original analysis approach. 

* That is, you can just apply your original analysis method to each of the $K$ complete datasets.

* The complicated part in multiple imputation is generating the $K$ complete datasets in a **"valid"** way.

* Luckily, there are a number of **R** packages that implement different ways of creating
the $K$ **imputed datasets**.

### Multiple imputation with mice

* I will primarily focus on the **mice** package.
    + **mice** stands for **"Multivariate Imputation by Chained Equations"**

* The `mice` function within the **mice** package is the primary function
for performing **multiple imputation**.

* To use `mice`, just use `mice(df)` where `df` is the **name** of the dataframe.
    + (Set `print = FALSE` if you don't want it to print out the number of the iteration).
    + Choose a value of `seed` so that the results are reproducible.

```r
library(mice)
imputed.airs <- mice(airquality, print=FALSE, seed=101)
```

---

* The object returned by `mice` will have a component called `imp` which is a list.
    
* Each component of `imp` is a **dataframe** corresponding to a single variable in the original dataframe 
     + This dataframe will contain the imputed values for the **missing values** of that variable.      

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
   + This is because the `Ozone` variable had **7 missing values**, and there are **5 multiple imputations**.


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

---


## Different Missing Data Mechanisms

### Missing Completely at Random (MCAR)

### Missing at Random (MAR)

### Missing not at Random (MNAR)





