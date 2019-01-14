# Author: Begum Topcuoglu
# Date: 2019-01-14
######################################################################
# This function defines the hyper-parameter budget of each classifier.
######################################################################
# Define the tuning grid for each of the seven classifiers as well as the cross-validation and classification algorithm. 

# Models we can choose from are:
# models = c("L2_Logistic_Regression", "L2_Linear_SVM", "RBF_SVM", "Decision_Tree", "Random_Forest","XGBoost")
######################################################################

tuning_grid <- function(model){
  
  # Cross-validation method
  cv <- trainControl(method="repeatedcv",
                     repeats = 10,
                     number=5,
                     returnResamp="final",
                     classProbs=TRUE,
                     summaryFunction=twoClassSummary,
                     indexFinal=NULL,
                     savePredictions = TRUE)
  
  if(model=="L2_Logistic_Regression") {
    grid <-  expand.grid(cost = c(0.5, 0.6, 0.7, 0.8, 0.9, 1),
                         loss = "L2_dual",
                         epsilon = 0.1)
    method <- "regLogistic"
  }
  else if (model=="L2_Linear_SVM"){
    grid <- expand.grid(C = c(0.015, 0.025, 0.035, 0.05, 0.06))
    method <- "svmLinear"
  }
  else if (model=="RBF_SVM"){
    grid <-  expand.grid(sigma = c(0.00000001, 0.0000001, 0.000001, 0.00001),
                         C = c(0.000001, 0.00001, 0.0001, 0.001))
    method <-"svmRadial"
  }
  else if (model=="Decision_Tree"){
    grid <-  expand.grid(maxdepth = c(1,2,3,4,5,6,7,8,9,10))
    method <-"rpart2"
  }
  else if (model=="Random_Forest"){
    grid <-  expand.grid(mtry = c(10,80,500,1000,1500,2000))
    method = "rf"
  }
  else if (model=="XGBoost"){
    grid <-  expand.grid(nrounds=100,
                         gamma=0,
                         eta=c(0.01, 0.1),
                         max_depth=c(6,7,8),
                         colsample_bytree= 0.8,
                         min_child_weight=c(1,2,3),
                         subsample=c(0.7,0.8,0.9))
    method <- "xgbTree"
  }
  else { 
    print("Model not available")
  }
  params <- list(grid, method, cv)
  return(params)
}

