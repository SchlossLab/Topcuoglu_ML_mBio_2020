library(tidyverse)
library(caret)
library(pROC)

######################## DATA PREPARATION #############################
# Features: Hemoglobin levels and 16S rRNA gene sequences in the stool
# Labels: - Colorectal lesions of 490 patients.
#         - Defined as cancer or not.(Cancer here means: SRN)
# Read in metadata and select only sample Id and diagnosis columns
meta <- read.delim('data/metadata.tsv', header=T, sep='\t') %>%
  select(sample, Dx_Bin, fit_result)
# Read in OTU table and remove label and numOtus columns
shared <- read.delim('data/baxter.0.03.subsample.shared', header=T, sep='\t') %>%
  select(-label, -numOtus)
# Merge metadata and OTU table.
# Group advanced adenomas and cancers together as cancer and normal, high risk normal and non-advanced adenomas as normal
# Then remove the sample ID column
data <- inner_join(meta, shared, by=c("sample"="Group")) %>%
  mutate(dx = case_when(
    Dx_Bin== "Adenoma" ~ "normal",
    Dx_Bin== "Normal" ~ "normal",
    Dx_Bin== "High Risk Normal" ~ "normal",
    Dx_Bin== "adv Adenoma" ~ "cancer",
    Dx_Bin== "Cancer" ~ "cancer"
  )) %>%
  select(-sample, -Dx_Bin) %>%
  drop_na()
# We want the diagnosis column to a factor
data$dx <- factor(data$dx, labels=c("normal", "cancer"))
inTraining <- createDataPartition(data$dx, p = .80, list = FALSE)
training <- data[ inTraining,]
testing  <- data[-inTraining,]
# Scale all features between 0-1
preProcValues <- preProcess(training, method = "range")
trainTransformed <- predict(preProcValues, training)
x_train <- trainTransformed %>% 
  select(-dx)
y_train <- trainTransformed %>% 
  select(dx)
testTransformed <- predict(preProcValues, testing)
x_test <- testTransformed %>% 
  select(-dx)
y_test <- testTransformed %>% 
  select(dx)
library(LiblineaR)
#########################################################################
model <- LiblineaR(x_train, trainTransformed$dx, type = 5)
pred=predict(model,x_test, decisionValues = TRUE)
# https://www.csie.ntu.edu.tw/~cjlin/papers/plattprob.pdf
deci <- pred[[2]]
label <- ifelse(y_test == "cancer", TRUE, FALSE)
prior1 <- sum(label==TRUE)
prior0 <- sum(label==FALSE)
#Parameter setting
maxiter <- 100 # Maximum number of iterations
minstep <-1e-10 # Minimum step taken in line search
sigma <- 1e-12 # Set to any value > 0
#Construct initial values: target support in array t,
# initial function value in fval
hiTarget <- (prior1+1.0)/(prior1+2.0)
loTarget <- 1/(prior0+2.0)
len=prior1+prior0 # Total number of data
t <- c()
for(i in 1:len){
  if(label[i] > 0){
    t[i] <- hiTarget}
  else{
    t[i] <- loTarget}
}

A <- 0.0 
B <- log((prior0+1.0)/(prior1+1.0))
fval <- 0.0

for(i in 1:len){
  
  fApB <-  deci[i]*A+B
  
  if(fApB >= 0){
    fval <- fval + t[i]*fApB+log(1+exp(-fApB))
    }
  else{
    fval <-  fval + (t[i]-1)*fApB+log(1+exp(fApB))
    }
}

for(it in 1:maxiter){
  ##Update Gradient and Hessian (use H’ = H + sigma I)
  h11=h22=sigma
  h21=g1=g2=0.0
  for(i in 1:len){
    fApB <- deci[i]*A+B
      if (fApB >= 0){
        p=exp(-fApB)/(1.0+exp(-fApB)) 
        q=1.0/(1.0+exp(-fApB))}
      else{
        p=1.0/(1.0+exp(fApB))
        q=exp(fApB)/(1.0+exp(fApB))}
    d2=p*q
    h11 <- h11 + deci[i]*deci[i]*d2
    h22 <- h22 + d2
    h21 <- h21 + deci[i]*d2
    d1 <- t[i]-p
    g1 <- g1 + deci[i]*d1
    g2 <- g1 + d1
  }
  if (abs(g1)<1e-5 && abs(g2)<1e-5) break##Stopping criteria

##Compute modified Newton directions
det=h11*h22-h21*h21
dA=-(h22*g1-h21*g2)/det
dB=-(-h21*g1+h11*g2)/det
gd=g1*dA+g2*dB
stepsize=1

while (stepsize >= minstep){ ##Line search
  newA=A+stepsize*dA, newB=B+stepsize*dB, newf=0.0
  for i = 1 to len {
    fApB=deci[i]*newA+newB
    if (fApB >= 0)
      newf += t[i]*fApB+log(1+exp(-fApB))
      else
        newf += (t[i]-1)*fApB+log(1+exp(fApB))
  }