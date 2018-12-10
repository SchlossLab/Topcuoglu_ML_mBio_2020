# Author: Begum Topcuoglu
# Date: 2018-12-05
#
######################################################################
# This script plots the cv and test AUC values as a boxplot for L2 Linear SVM classification model. It also plots the cvAUC values for each hyper-parameter that us tuned.
######################################################################



######################################################################
#-----------Read in necessary libraries and functions ---------------#
######################################################################
deps = c("reshape2", "cowplot", "ggplot2","knitr","rmarkdown","vegan","gtools", "tidyverse");
for (dep in deps){
  if (dep %in% installed.packages()[,"Package"] == FALSE){
    install.packages(as.character(dep), quiet=TRUE);
  }
  library(dep, verbose=FALSE, character.only=TRUE)
}

# Load in needed functions
source('code/learning/functions.R')

######################################################################
# Load .tsv data generated with modeling pipeline for Logistic Regression
######################################################################

# Read in the cvAUCs, testAUCs and hyper-parameters from L2 logistic regression model for 100 splits.
l2svm <- read.delim('data/process/L2_Linear_SVM_aucs_hps_R.tsv', header=T, sep='\t')

# Exploratory look at the numbers
l2svm %>%
  summarise(cv_mean=mean(cv_aucs), 
            test_mean=mean(test_aucs), 
            cv_sd=sd(cv_aucs), 
            test_sd=sd(test_aucs))

######################################################################
#Plot the AUC values for cross validation and testing for each model #
######################################################################

performance <- plot_performance(logit)

######################################################################
#Plot the cvAUC values for each hyperparameter in the budget #
######################################################################

parameter <- plot_parameter(logit)

######################################################################
#-------------------- Put the plots together -----------------#
######################################################################

plot_grid(performance, parameter, labels = c("A", "B"))

######################################################################
#-----------------------Save figure as .pdf ------------------------ #
######################################################################

ggsave("L2_Linear_SVM_comparison.pdf", plot = last_plot(), device = 'pdf', path = 'results/figures', width = 15, height = 10)

