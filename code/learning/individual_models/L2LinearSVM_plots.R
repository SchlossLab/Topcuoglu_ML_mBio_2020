######################################################################
#----------------- Read in necessary libraries -------------------#
######################################################################
deps = c("reshape2", "cowplot", "ggplot2","knitr","rmarkdown","vegan","gtools", "tidyverse");
for (dep in deps){
  if (dep %in% installed.packages()[,"Package"] == FALSE){
    install.packages(as.character(dep), quiet=TRUE);
  }
  library(dep, verbose=FALSE, character.only=TRUE)
}

# Load in needed functions and libraries
source('code/learning/functions.R')

# Read in hyper-parameter AUCs generated from L2 logistic regression model for all samples:
l2svm <- read.delim('data/process/L2_Linear_SVM_aucs_hps_R.tsv', header=T, sep='\t')

l2svm %>%
  summarise(cv_mean=mean(cv_aucs), 
            test_mean=mean(test_aucs), 
            cv_sd=sd(cv_aucs), 
            test_sd=sd(test_aucs))

performance <- plot_performance(l2svm)
parameter <- plot_parameter(l2svm)

plot_grid(performance, parameter, labels = c("A", "B"))



######################################################################
#-----------------------Save figure as .pdf ------------------------ #
######################################################################

ggsave("L2_Linear_SVM_comparison.pdf", plot = last_plot(), device = 'pdf', path = 'results/figures', width = 15, height = 10)

