# Missing Data and Multiple Imputation {#missing-data}

## Missing Data in R and "Naive Approaches"

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
sum( is.na(airquality) )  ## 44 missing observations in total
```

```
## [1] 44
```

```r
dim( airquality )  ## Dataset has a total of 153 x 6 = 918 observations
```

```
## [1] 153   6
```

---


* Complete Case Analysis or Listwise Deletion.

## Multiple Imputation

* To summarize, **multiple imputation** consists of the following steps:
     1. Create $K$ different "complete datasets" which contain no missing data.
     
     2. For each of these $K$ complete datasets, compute the estimates of interest. 
     
     3. "Pool" these separate estimates to get a final estimate. 

---

* The nice thing about multiple imputation is that you can always
use your original analysis approach. 
    + That is, you can just apply your original analysis method to each of the $K$ complete datasets.

* The tricky part in multiple imputation is generating the $K$ complete datasets in a "valid" way.

---


## Different Missing Data Mechanisms

### Missing Completely at Random (MCAR)

### Missing at Random (MAR)

### Missing not at Random (MNAR)





