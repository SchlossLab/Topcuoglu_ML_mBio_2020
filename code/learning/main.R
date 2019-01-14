#### Author: Begum Topcuoglu
#### Date: 2018-12-20
#### Title: Main pipeline for 7 classifiers in R programming language
######################################################################
#### Description: This script will read in 0.03 subsampled OTU dataset and the metadata that has the cancer diagnosis. 

#### It will run L2 Logistic Regression, L1 and L2 Linear SVMs, RBF SVM, Decision Tree, Random Forest and XGBoost classifiers
######################################################################
#### To be able to run this script we need to be in our project directory.

#### The outputs are (1) AUC values for cross-validation and testing for each data-split (2) meanAUC values for each hyper-parameter tested during each split.
######################################################################


############################# IMPORT LIBRARIES ##################################
# The dependinces for this script are consolidated in the first part
deps = c("reshape2", "kernlab","LiblineaR", "doParallel","pROC", "caret", "gtools", "tidyverse");
for (dep in deps){
  if (dep %in% installed.packages()[,"Package"] == FALSE){
    install.packages(as.character(dep), quiet=TRUE, repos = "http://cran.us.r-project.org");
  }
  library(dep, verbose=FALSE, character.only=TRUE)
}

# Load in needed functions and libraries
source('code/learning/functions.R')
source('code/learning/model_selection.R')
source('code/learning/model_pipeline.R')
##############################################################################


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

# Save results of the modeling pipeline as a list
results <- pipeline(data, "L2_Logistic_Regression") 

# Create a matrix with cv_aucs and test_aucs from 100 data splits
full <- matrix(c(results[[1]], results[[2]]), ncol=2) 

# Convert to dataframe and add a column noting the model name
data <- data.frame(full) %>% 
  rename(cv_aucs=X1, test_aucs=X2) %>% 
  mutate(model="L2 Logistic Regression") 

# Box-plot performance for cross-validation and testing AUCs values
plot_performance(data)






