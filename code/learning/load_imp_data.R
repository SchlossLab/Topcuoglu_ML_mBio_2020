######################################################################
#----------------- Read in necessary libraries -------------------#
######################################################################
deps = c("reshape2","knitr","rmarkdown","vegan","gtools", "tidyverse");
for (dep in deps){
  if (dep %in% installed.packages()[,"Package"] == FALSE){
    install.packages(as.character(dep), quiet=TRUE);
  }
  library(dep, verbose=FALSE, character.only=TRUE)
}

######################################################################
#----------------- Load feature importance data -----------------#
######################################################################
# Read in non-linear model's base test AUC median

rbf_median <- read.delim('../data/process/RBF_SVM_non_linear_base_median.tsv', header=T, sep='\t')$. 
dt_median <- read.delim('../data/process/Decision_Tree_non_linear_base_median.tsv', header=T, sep='\t')$. 
rf_median <- read.delim('../data/process/Random_Forest_non_linear_base_median.tsv', header=T, sep='\t')$. 
xgboost_median <- read.delim('../data/process/XGBoost_non_linear_base_median.tsv', header=T, sep='\t')$. 

# Read in non-linear model's top 5 feature importances
rbf_imp <- read.delim('../data/process/RBF_SVM_non_linear_top_five_importance.tsv', header=T, sep='\t')
dt_imp <- read.delim('../data/process/Decision_Tree_non_linear_top_five_importance.tsv', header=T, sep='\t')
rf_imp <- read.delim('../data/process/Random_Forest_non_linear_top_five_importance.tsv', header=T, sep='\t')
xgboost_imp <- read.delim('../data/process/XGBoost_non_linear_top_five_importance.tsv', header=T, sep='\t')