
# London School of Economics and Political Science

# Machine Learning Coursework Project

library(rpart)
library(rpart.plot)
library(nnet)
library(randomForest)

#Notes
#limitations on data
#unaware how the data was collected = could lead to biased/ collected unfairly with malicious intent
#write about how in the past there was documented gender discrimination and the change that has occurred.

#Data set: https://www.kaggle.com/datasets/hjmjerry/gender-discrimination?select=Lawsuit.csv
#API Command: kaggle datasets download -d hjmjerry/gender-discrimination

getwd()
setwd()
data1 <- read.csv("Lawsuit.csv")

# Scatter plot Matrix with smooth curves
pairs(~ . , panel=panel.smooth, data = data1, main = "Scatterplot Matrix of Lawsuit Data")
## Sal94 and Sal95 have very high correlation.
## Promotion info not in data.
## rank info available
## Some Dept has consistently high salary. such as surgery 
## Fewer Females than Males and top Salaries belong to Males.
## Top salaries with Clinical Emphasis.
## Top salaries with Board Certified.
## doctors who do more research earn less = could be because of less patients

## No information on Promotion. Thus focus on Sal94. Sal95 is highly correlated.


# Discard ID and Sal95 in analysis
data2 <- data1[,2:9]


# (Principle Component Analysis) PCA ------------------------------------------
pc <- prcomp(data2, scale.=T) 
summary(pc)
## First two principal components capture 70% of variance. 
pc$rotation
## Gender is relatively not important in PC1 but relatively important in PC2.
## PCA concludes Gender is important differentiater.


data2.scaled <- scale(data2)


# K Means Clustering -----------------------------------------------------------
set.seed(2020)
k2 <- kmeans(data2.scaled, centers=2)  
# set k = 2 to see natural clusters of 2.
summary(k2)
k2results <- data.frame(data2$Gender, data2$Sal94, k2$cluster)
cluster1 <- subset(k2results, k2$cluster==1)
cluster2 <- subset(k2results, k2$cluster==2)
cluster1$data2.Gender <- factor(cluster1$data2.Gender)
cluster2$data2.Gender <- factor(cluster2$data2.Gender)
#GD affects my pay
#GD affects my promotion hence two clusters
#look at gender ratio for two clusters, which one is higher in terms of gender

summary(cluster1$data2.Sal94)
summary(cluster2$data2.Sal94)
## Cluster 1 has higher salary than cluster 2.


round(prop.table(table(cluster1$data2.Gender)),2)
round(prop.table(table(cluster2$data2.Gender)),2)
## 67% in Cluster 1 are Males, 50% in Cluster 2 are Males.


# Goodness of Fit Test
# Is Cluster 1 statistically same as Cluster 2 in terms of Gender?
M <- as.matrix(table(cluster1$data2.Gender))
p.null <- as.vector(prop.table(table(cluster2$data2.Gender)))
chisq.test(M, p=p.null)
## Cluster 1 Gender Proportions are different from Cluster 2 Gender Proportions
## K-means clustering concludes Gender is significant differentiator.
# cluster 1 has more males = higher salary


# Hierarchical Clustering Results----------------------------------------------------
hc.average =hclust(dist(data2.scaled), method ="average")

plot(hc.average , main ="Average Linkage", xlab="", sub ="", cex =.9)
sum(cutree(hc.average , 2)==2)  ## 2
## Average linkage fails to provide sufficient sample size for one cluster.
# Dominated by one cluster

hc.complete =hclust(dist(data2.scaled), method ="complete")
plot(hc.complete , main ="Complete Linkage", xlab="", sub ="", cex =.9)
sum(cutree(hc.complete, 2)==2)  ## 159 cases of second cluster
hc.cluster1 <- subset(k2results, cutree(hc.complete, 2)==1)
hc.cluster2 <- subset(k2results, cutree(hc.complete, 2)==2)
# Complete linkage works better, you have sufficient sample size for two samples.


hc.cluster1$data2.Gender <- factor(hc.cluster1$data2.Gender)
hc.cluster2$data2.Gender <- factor(hc.cluster2$data2.Gender)

summary(hc.cluster1$data2.Sal94)
summary(hc.cluster2$data2.Sal94)
## Cluster 2 has higher salary than cluster 1.


round(prop.table(table(hc.cluster1$data2.Gender)),2)
round(prop.table(table(hc.cluster2$data2.Gender)),2)
## 62% in Cluster 2 are Males, 55% in Cluster 1 are Males.

# Goodness of Fit Test
# Is hc.cluster 2 statistically same as hc.cluster 1 in terms of Gender?
M <- as.matrix(table(hc.cluster2$data2.Gender))
p.null <- as.vector(prop.table(table(hc.cluster1$data2.Gender)))
chisq.test(M, p=p.null)
## Cluster 2 Gender Proportions are similar statistically from Cluster 1 Gender Proportions
## Hierarchical Clustering concludes Gender is insignificant differentiator between the 2 clusters.
# they are statistically similar. You cannot reject the null hypothesis that the distribution are the same
# the few percentage points can be attributed to random chance
# chi test shows its not significant

# Linear Regression Results

# Linear Regression with Sal94 outcome -----------------------------------------
data2.dum <- data2
data2.dum$Dept <- factor(data2.dum$Dept)
data2.dum$Gender <- factor(data2.dum$Gender)
data2.dum$Clin <- factor(data2.dum$Clin)
data2.dum$Cert <- factor(data2.dum$Cert)
data2.dum$Rank <- factor(data2.dum$Rank)
# dummy variables in the data set
# factoring in dept, gender etc

m.lin <- lm(Sal94 ~ ., data = data2.dum)
rmse.linreg <- round(sqrt(mean(residuals(m.lin)^2)),0)
summary(m.lin)
## Gender is stat insignificant. Dept, Cert, Exper, Clin, Rank are sig.
# "." meaning all the other variables
# the model can also predict the salary 
# traditional way is checking the x variable is significant
# gender has no * = not a significant predictor


# CART with Sal94 outcome --------------------------------

set.seed(2)
m.cart <- rpart(Sal94 ~ ., data = data2.dum, method = 'anova', control = rpart.control(minsplit = 2, cp = 0))
printcp(m.cart)
plotcp(m.cart)
# using anova to do a regression tree
# minimum to split = 2
# tree is grown to the max
# xerror is going down rapidly = less than 1 = effective

# Extract the Optimal Tree via code ------------
# Compute min CVerror + 1SE in maximal tree m.cart.
CVerror.cap <- m.cart$cptable[which.min(m.cart$cptable[,"xerror"]), "xerror"] + m.cart$cptable[which.min(m.cart$cptable[,"xerror"]), "xstd"]
#used to predict salary = does not overfit or underfit

# Find the optimal CP region whose CV error is just below CVerror.cap in maximal tree m.cart.
i <- 1; j<- 4
while (m.cart$cptable[i,j] > CVerror.cap) {
  i <- i + 1
}

# Get the geometric mean of the two identified CP values in the optimal region if optimal tree has at least one split.
cp.opt = ifelse(i > 1, sqrt(m.cart$cptable[i,1] * m.cart$cptable[i-1,1]), 1)
# ------------------------------------------------------------------------------

## i = 13 shows that the 13th tree is optimal based on 1 SE rule.

# Prune the max tree using a particular CP value
m.cart2 <- prune(m.cart, cp = cp.opt)

rmse.cart1 <- round(sqrt(mean((data2.dum$Sal94 - predict(m.cart))^2)),0)

rmse.cart2 <- round(sqrt(mean((data2.dum$Sal94 - predict(m.cart2))^2)),0)

## Unpruned CART has over fitted the train set but we are not predicting future cases but only explaining historical data.

m.cart$variable.importance
## Dept, Prate, Clin, Experience, Board-Certified are more important than Gender in explaining Salary. 
#department is the most important gauge for your salary according to the maximum tree
# Results: Gender is not significant predictor of Salary. Hence, no gender discrimination on Salary.


# Logistic Regression and CART---------------------------------------------------


# Logistic Reg. Is Distribution of rank associated statistically with Gender in the absence of salary?
m.log <- multinom(Rank ~ . -Sal94 , data = data2.dum)
summary(m.log)
OR.CI <- exp(confint(m.log))
OR.CI
# odds ratio CI
# salary is subtracted = salary is high rank = you get promoted first then salary is given
# salary is a big indicator of rank
# odds ratio is a fraction
# if the interval captures 1
# 1.226 to 7.17 so gender is statically significant
## Gender is statistically significant in distribution of rank but
## does not necessarily mean promotion discriminatory.
## hiring decision could be the cause of this discrepancy.

# CART on Rank ----------------------------------------------------------------
set.seed(2)
m.cart.rk <- rpart(Rank ~ . -Sal94, data = data2.dum, method = 'class', control = rpart.control(minsplit = 2, cp = 0))
printcp(m.cart.rk)
plotcp(m.cart.rk)

# Using the maximal tree as objective is explaining historical data, not predicting future cases.
m.cart.rk$variable.importance
# Experience, Prate and Dept are more important than Gender in explaining Rank.
# 1st model is saying gender is important, 2nd model is saying its not so important. Perhaps only marginally.
# why don't we split with train and testing = used to predict the future. we are not predicting salary or rank.


## Conclusions:
## Insufficient evidence of Gender Discrimination on salary.
## No information to determine Promotion bias as only the current rank is given.
## Evidence of Gender Discrimination on Rank is mixed. i.e. not conclusive.
## Dept and Experience are far more important than Gender.

