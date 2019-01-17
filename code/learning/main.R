######################################################################
# Author: Begum Topcuoglu
# Date: 2018-12-20
# Title: Main pipeline for 7 classifiers in R programming language
######################################################################

######################################################################
# Description: 

# This script will read in data from Baxter et al. 2016
#     - 0.03 subsampled OTU dataset
#     - CRC metadata: SRN information


# It will run the following machine learning pipelines:
#     - L2 Logistic Regression 
#     - L1 and L2 Linear SVM
#     - RBF SVM
#     - Decision Tree
#     - Random Forest 
#     - XGBoost 
######################################################################

######################################################################
# Dependencies and Outputs: 

# Be in the project directory.

# The outputs are:
#   (1) AUC values for cross-validation and testing for each data-split 
#   (2) meanAUC values for each hyper-parameter tested during each split.
######################################################################


################### IMPORT LIBRARIES and FUNCTIONS ###################
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
source('code/learning/generateAUCs.R')
######################################################################

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
###################################################################

######################## RUN PIPELINE #############################
# Choose which classification methods we want to run
model_names = c("L2_Logistic_Regression", 
                "L2_Linear_SVM", 
                "RBF_SVM", 
                "Decision_Tree", 
                "Random_Forest",
                "XGBoost")
# Get the cv and test AUCs for 100 data-splits
model <- as.character(commandArgs(TRUE)) # recieve input from model
getAucs(model)
###################################################################




