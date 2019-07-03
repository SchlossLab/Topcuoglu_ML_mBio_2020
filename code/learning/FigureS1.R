# Author: Begum Topcuoglu
# Date: 2018-02-12
#
######################################################################
# This script plots Figure 2:
#   1. Y axis: mean cvAUC of 100 datasplits
#   2. X axis: different hyper-parameters tested in cv(hp)
######################################################################


######################################################################
# Load in needed functions and libraries
source('code/learning/functions.R')
# detach("package:randomForest", unload=TRUE) to run
######################################################################
#----------------- Read in necessary libraries -------------------#
######################################################################
deps = c("scales","cowplot", "ggplot2","knitr","rmarkdown","vegan","gtools", "tidyverse");
for (dep in deps){
  if (dep %in% installed.packages()[,"Package"] == FALSE){
    install.packages(as.character(dep), quiet=TRUE);
  }
  library(dep, verbose=FALSE, character.only=TRUE)
}
######################################################################


######################################################################
# Load .tsv data generated with modeling pipeline for Logistic Regression
######################################################################

# Read in the results of trained model of 100 data-splits

all_files <- list.files(path= 'data/process', pattern='combined_all_hp.*', full.names = TRUE)

logit_all <- read_files(all_files[4])
l2svm_all <- read_files(all_files[3])
l1svm_all <- read_files(all_files[2])

######################################################################
#Plot the mean AUC values for hyper parameters tested #
######################################################################

# Define the base plot for all the modeling methods
base_plot <-  function(data, x_axis, y_axis){
  plot <- ggplot(data, aes(x_axis, y_axis)) +
    geom_line() +
    geom_point() +
    theme_bw() +
    geom_hline(yintercept = 0.5, linetype="dashed") +
    theme(legend.text=element_text(size=10),
          legend.title=element_text(size=10),
          panel.grid.major = element_blank(),
          panel.grid.minor = element_blank(),
          panel.background = element_blank(),
          text = element_text(size = 10),
          axis.text.x=element_text(size = 8, colour='black'),
          axis.text.y=element_text(size = 8, colour='black'),
          axis.title.y=element_text(size = 10),
          axis.title.x=element_text(size = 10))
  return(plot)
}

# Start plotting models with one hyper-parameter individually
l1svm <- l1svm_all %>%
  group_by(cost) %>%
  summarise(mean_AUC = mean(ROC), sd_AUC = sd(ROC))

l1svm_plot <- base_plot(l1svm, l1svm$cost, l1svm$mean_AUC) +
  scale_x_log10(name="C (penalty)",
                labels=trans_format('log10',math_format(10^.x))) +
  scale_y_continuous(name="L1 linear SVM
                     mean cvAUROC",
                     limits = c(0.30, 1),
                     breaks = seq(0.3, 1, 0.1)) +
  geom_errorbar(aes(ymin=mean_AUC-sd_AUC, ymax=mean_AUC+sd_AUC), width=.001)


l2svm <- l2svm_all %>%
  group_by(cost) %>%
  summarise(mean_AUC = mean(ROC), sd_AUC = sd(ROC))

l2svm_plot <- base_plot(l2svm, l2svm$cost, l2svm$mean_AUC) +
  scale_x_log10(name="C (penalty)",
                labels=trans_format('log10',math_format(10^.x))) +
  scale_y_continuous(name="L2 linear SVM
                     mean cvAUROC",
                     limits = c(0.30, 1),
                     breaks = seq(0.3, 1, 0.1)) +
  geom_errorbar(aes(ymin=mean_AUC-sd_AUC, ymax=mean_AUC+sd_AUC), width=.001)

logit <- logit_all %>%
  group_by(cost, loss, epsilon) %>%
  summarise(mean_AUC = mean(ROC), sd_AUC = sd(ROC))

logit_plot <- base_plot(logit, logit$cost, logit$mean_AUC) +
  scale_x_log10(name="C (penalty)",
                labels=trans_format('log10',math_format(10^.x))) +
  scale_y_continuous(name="L2 logistic regression
                     mean cvAUROC",
                     limits = c(0.30, 1),
                     breaks = seq(0.3, 1, 0.1)) +
  geom_errorbar(aes(ymin=mean_AUC-sd_AUC, ymax=mean_AUC+sd_AUC), width=.001)


######################################################################
#-----------------------Save figure as .pdf ------------------------ #
######################################################################

ggsave("Figure_S1.png", plot = linear_models, device = 'png', path = 'submission', width = 8, height = 2.5)

