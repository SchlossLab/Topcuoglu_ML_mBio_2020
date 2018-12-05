#### Author: Begum Topcuoglu
#### Date: 2018-10-11
#### Title: L1 Linear SVM Pipeline for Baxter GLNE007 Dataset

#### Description: This script will read in 0.03 subsampled OTU dataset and the metadata that has the cancer diagnosis. It generates a L1 support vector machine model. The model is trained on 80% of the data and then tested on 20% of the data. It also plots the cross validation and testing ROC curves to look at generalization performance of the model.

#### To be able to run this script we need to be in our project directory.

#### The dependinces for this script are consolidated in the first part
deps = c("kernlab","LiblineaR", "doParallel","pROC", "caret", "gtools", "tidyverse");
for (dep in deps){
  if (dep %in% installed.packages()[,"Package"] == FALSE){
    install.packages(as.character(dep), quiet=TRUE, repos = "http://cran.us.r-project.org");
  }
  library(dep, verbose=FALSE, character.only=TRUE)
}

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

# Create
best.tunes <- c()
all.test.response <- all.test.predictor <- test_aucs <- c()
all.cv.response <- all.cv.predictor <- cv_aucs <- c()
cl <- makePSOCKcluster(4)
registerDoParallel(cl)
for (i in 1:100) {
  inTraining <- createDataPartition(data$dx, p = .80, list = FALSE)
  training <- data[ inTraining,]
  testing  <- data[-inTraining,]
  preProcValues <- preProcess(training, method = "range")
  trainTransformed <- predict(preProcValues, training)
  testTransformed <- predict(preProcValues, testing)
  grid <- expand.grid(C = c(0.015, 0.025, 0.035, 0.05, 0.06))
  cv <- trainControl(method="repeatedcv",
                     repeats = 10,
                     number=5,
                     returnResamp="final",
                     classProbs=TRUE,
                     summaryFunction=twoClassSummary,
                     indexFinal=NULL,
                     savePredictions = TRUE)
  
  L2SVM <- train(dx ~ .,
                  data=trainTransformed,
                   method = "svmLinear",
                   trControl = cv,
                   metric = "ROC",
                 tuneGrid = grid)
  
  # Best lambda parameter
  best.tune <- L2SVM$bestTune[1]
  best.tunes <- c(best.tunes, best.tune)
  print(L2SVM$bestTune)
  # Mean AUC value of the best lambda parameter training over repeats
  print(max(L2SVM$results[,"ROC"]))
  cv_auc <- getTrainPerf(L2SVM)$TrainROC
  # Plot parameter performane
  #trellis.par.set(caretTheme())
  #plot(L2LogicalRegression)
  # Predict on the test set and get predicted probabilities
  rpartProbs <- predict(L2SVM, testTransformed, type="prob")
  # Test AUC calculation
  test_roc <- roc(ifelse(testTransformed$dx == "cancer", 1, 0), rpartProbs[[2]])
  test_auc <- test_roc$auc
  print(test_auc)
  # Save all the test AUCs over iterations in test_aucs
  test_aucs <- c(test_aucs, test_auc)
  # Cross-validation mean AUC value
  # Save all the test AUCs over iterations in cv_aucs
  cv_aucs <- c(cv_aucs, cv_auc)
  # Save the test set labels in all.test.response. Labels converted to 0 for normal and 1 for cancer
  all.test.response <- c(all.test.response, ifelse(testTransformed$dx == "cancer", 1, 0))
  # Save the test set predicted probabilities of highest class in all.test.predictor
  all.test.predictor <- c(all.test.predictor, rpartProbs[[2]])
  # Save the training set labels in all.test.response. Labels are in the obs column in the training object
  #all.cv.response <- c(all.cv.response, L2Logit$pred$obs)
  # Save the training set labels
  #all.cv.predictor <- c(all.cv.predictor, L2Logit$pred$normal)
}
on.exit(stopCluster(cl))
# Get the ROC of both test and cv from all the iterations
test_roc <- roc(all.test.response, all.test.predictor, auc=TRUE, ci=TRUE)
#cv_roc <- roc(all.cv.response, all.cv.predictor, auc=TRUE, ci=TRUE)
full <- matrix(c(cv_aucs, test_aucs, best.tunes), ncol=3)
write.table(full, file='data/process/L2_Linear_SVM_aucs_hps_R.tsv', quote=FALSE, sep='\t', col.names = c("cv_aucs","test_aucs", "Cost"), row.names = FALSE)

pdf("results/figures/L2_Linear_SVM_inR.pdf")
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
