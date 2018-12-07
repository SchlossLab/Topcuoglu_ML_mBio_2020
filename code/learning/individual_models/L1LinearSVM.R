#### Author: Begum Topcuoglu
#### Date: 2018-10-11
#### Title: L1 Linear SVM Pipeline for Baxter GLNE007 Dataset

#### Description: This script will read in 0.03 subsampled OTU dataset and the metadata that has the cancer diagnosis. It generates a L1 support vector machine model. The model is trained on 80% of the data and then tested on 20% of the data. It also plots the cross validation and testing ROC curves to look at generalization performance of the model.

#### To be able to run this script we need to be in our project directory.

#### The dependinces for this script are consolidated in the first part
deps = c("LiblineaR", "doParallel","pROC", "caret","knitr","rmarkdown","vegan","gtools", "tidyverse");
for (dep in deps){
  if (dep %in% installed.packages()[,"Package"] == FALSE){
    install.packages(as.character(dep), quiet=TRUE);
  }
  library(dep, verbose=FALSE, character.only=TRUE)
}

# Read in metadata and select only sample Id and diagnosis columns
meta <- read.delim('data/metadata.tsv', header=T, sep='\t') %>%
  select(sample, dx)

# Read in OTU table and remove label and numOtus columns
shared <- read.delim('data/baxter.0.03.subsample.shared', header=T, sep='\t') %>%
  select(-label, -numOtus)

# Merge metadata and OTU table and remove all the samples that are diagnosed with adenomas. Keep only cancer and normal.
# Then remove the sample ID column
data <- inner_join(meta, shared, by=c("sample"="Group")) %>%
  filter(dx != 'adenoma') %>%
  select(-sample)

# We want the diagnosis column to a factor
data$dx <- factor(data$dx, labels=c("normal", "cancer"))

all.test.response <- all.test.predictor <- c()
all.cv.response <- all.cv.predictor <- c()
cl <- makePSOCKcluster(4)
registerDoParallel(cl)
for (i in 1:5) {
  inTraining <- createDataPartition(data$dx, p = .80, list = FALSE)
  training <- data[ inTraining,]
  testing  <- data[-inTraining,]
  preProcValues <- preProcess(training, method = "range")
  training <- predict(preProcValues, training)
  testing <- predict(preProcValues, testing)
  x_train <- training %>% select(-dx)
  y_train <- ifelse(training$dx == "cancer", 1, 0)
  y_train <- as.factor(y_train)
  x_test <- testing %>% select(-dx)
  y_test <- testing$dx
  y_test <- ifelse(testing$dx == "cancer", 1, 0)
  y_test <- as.factor(y_test)
  # create LR classification model
  cParameterValues <- c(0.0002, 0.00025, 0.0003, 0.00035, 0.0004)
  bestCost=NA
  bestAcc=0
  for(co in cParameterValues){
    acc=LiblineaR(data=x_train,
                  target=y_train,
                  type=5,cost=co,
                  cross=5,
                  verbose=FALSE)
    cat("Results for C=",co," : ",acc," accuracy.\n",sep="")
    if(acc>bestAcc){
      bestCost=co
      bestAcc=acc
    }
  }
  cat("Best cost is:",bestCost,"\n")
  cat("Best accuracy is:",bestAcc,"\n")

  # Re-train best model with best cost value.
  model=LiblineaR(data=x_train, 
                  target=y_train, 
                  type=5, 
                  cost=bestCost, 
                  verbose=FALSE)
  
  classification <- predict(model, x_test, decisionValues = TRUE) 
  predictedLabels <- classification$predictions
  all.test.response <- c(all.test.response, y_test)
  # Save the test set predicted probabilities of highest class in all.test.predictor
  all.test.predictor <- c(all.test.predictor, classification$decisionValues[[2]])
}
stopCluster(cl)

test_roc <- roc(all.test.response, all.test.predictor, auc=TRUE, ci=TRUE)

pdf("results/figures/L1_Linear_SVM_inR.pdf")
par(mar=c(4,4,1,1))
# Plot random line on ROC curve
plot(c(1,0),c(0,1),
     type='l',
     lty=3,
     xlim=c(1.01,0), ylim=c(-0.01,1.01),
     xaxs='i', yaxs='i',
     ylab='', xlab='')
# Plot Test ROC in red line
plot(test_roc,
     col='black',
     lwd=2,
     add=T,
     lty=1)
# Compute the CI of the AUC
auc.ci <- ci.auc(test_roc)
# Plot CV ROC in blue line
#plot(cv_roc,
#     col='blue',
#     lwd=2,
#     add=T,
#     lty=1)
# Label the axes
mtext(side=2,
      text="Sensitivity",
      line=2.5,
      cex=1.5)
mtext(side=1,
      text="Specificity",
      line=2.5,
      cex=1.5)
# Add legends for both lines
legend(x=0.7,y=0.2,
       legend=(sprintf('Test - AUC: %.3g, CI: %.3g', test_roc$auc, (auc.ci[3]-auc.ci[2]))),
       bty='n',
       xjust=0,
       lty=c(1,1),
       col='black',
       text.col='black')


# Save the figure
dev.off()

