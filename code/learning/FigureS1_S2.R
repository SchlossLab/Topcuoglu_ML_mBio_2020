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
rbf_all <- read_files(all_files[6])
rf_all <- read_files(all_files[5])
dt_all <- read_files(all_files[1])
xgboost_all <- read_files(all_files[7])

######################################################################
#Plot the mean AUC values for hyper parameters tested #
######################################################################

# Define the base plot for all the modeling methods
base_plot <-  function(data, x_axis, y_axis){
  plot <- ggplot(data, aes(x_axis, y_axis)) +
  geom_line() +
  geom_point() +
  theme_bw() +
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
                     breaks = seq(0.3, 1, 0.05)) +
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
                     breaks = seq(0.3, 1, 0.05)) +
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
                     breaks = seq(0.3, 1, 0.05)) +
  geom_errorbar(aes(ymin=mean_AUC-sd_AUC, ymax=mean_AUC+sd_AUC), width=.001)


dt <- dt_all %>%
  group_by(maxdepth) %>%
  summarise(mean_AUC = mean(ROC), sd_AUC = sd(ROC))

dt_plot <- base_plot(dt, dt$maxdepth, dt$mean_AUC) +
scale_x_continuous(name="max depth") +
  scale_y_continuous(name="Decision tree
mean cvAUROC",
                     limits = c(0.30, 1),
                     breaks = seq(0.3, 1, 0.05)) +
  geom_errorbar(aes(ymin=mean_AUC-sd_AUC, ymax=mean_AUC+sd_AUC), width=.001)

rf <- rf_all %>%
  group_by(mtry) %>%
  summarise(mean_AUC = mean(ROC), sd_AUC = sd(ROC))

rf_plot <-  base_plot(rf, rf$mtry, rf$mean_AUC) +
scale_x_continuous(name="mtry",
                   breaks=seq(0, 1500, 250), limits = c(0, 1500)) +
  scale_y_continuous(name="Random forest
mean cvAUROC",
                     limits = c(0.30, 1),
                     breaks = seq(0.3, 1, 0.05)) +
  geom_errorbar(aes(ymin=mean_AUC-sd_AUC, ymax=mean_AUC+sd_AUC), width=1)

# Start plotting models with 2 hyper-parameters individually

rbf_data <- rbf_all %>%
  group_by(sigma, C) %>%
  summarise(mean_AUC = mean(ROC), sd_AUC = sd(ROC))

rbf_plot <- ggplot(rbf_data, aes(x = sigma, y = C, fill = mean_AUC)) +
  geom_tile() +
  scale_fill_gradient(name= "Mean cvAUROC",
                      low = "#FFFFFF",
                      high = "#012345") +
  #coord_fixed(ratio = 0.5) +
  #coord_equal() +
  scale_y_log10(name="SVM RBF kernel
  C",
                breaks = c(0.0000001, 0.000001, 0.00001, 0.0001, 0.001, 0.01, 0.1, 1, 10), 
                expand = c(0, 0), 
                labels=trans_format('log10',math_format(10^.x))) +
  scale_x_log10(breaks = c(0.00000001, 0.0000001, 0.000001, 0.00001, 0.0001, 0.001, 0.01, 0.1), 
                expand = c(0, 0), 
                labels=trans_format('log10',math_format(10^.x))) +
  theme(legend.background = element_rect(size=0.5, linetype="solid", color="black"),
        legend.box.margin=margin(c(1,1,1,1)),
        legend.text=element_text(size=6),
        legend.title=element_text(size=8), 
        legend.position="bottom",
        axis.title = element_text(size=10),
        axis.text = element_text(size=10),
        panel.border = element_rect(colour = "black", fill=NA, size=3), 
        panel.grid.major = element_blank(),
        panel.grid.minor = element_blank(),
        panel.background = element_blank(),
        axis.text.x=element_text(size = 8, colour='black'),
        axis.text.y=element_text(size = 8, colour='black'))

xgboost_data <- xgboost_all %>%
  group_by(eta, subsample) %>%
  summarise(mean_AUC = mean(ROC), sd_AUC = sd(ROC))

xgboost_plot <- ggplot(xgboost_data, aes(x = eta, y = subsample, fill = mean_AUC)) +
  geom_tile() +
  #coord_fixed(ratio = 5) +
  scale_fill_gradient(name= "Mean cvAUROC",
                      low = "#FFFFFF",
                      high = "#012345") +
  scale_y_continuous(name="XGBoost
subsample",
    breaks = c(0.4, 0.5, 0.6, 0.7),
    expand=c(0,0)) +
  scale_x_log10(breaks = c(0.001, 0.01, 0.1, 1),
                expand = c(0, 0),
                labels=trans_format('log10',math_format(10^.x))) +
  theme(axis.title = element_text(size=10),
        axis.text = element_text(size=10),
        panel.border = element_rect(colour = "black", fill=NA, size=3),
        panel.grid.major = element_blank(),
        panel.grid.minor = element_blank(),
        panel.background = element_blank(),
        axis.text.x=element_text(size = 8, colour='black'),
        axis.text.y=element_text(size = 8, colour='black'),
        legend.background = element_rect(size=0.5, linetype="solid", color="black"),
        legend.box.margin=margin(c(1,1,1,1)),
        legend.text=element_text(size=6),
        legend.title=element_text(size=8), legend.position="bottom")


linear_models <- plot_grid(logit_plot, l1svm_plot, l2svm_plot, labels = c("A", "B", "C"), ncol=3)

non_linear_models <- plot_grid(dt_plot, rf_plot, rbf_plot, xgboost_plot, labels = c("A", "B", "C", "D"), ncol=2)

######################################################################
#-----------------------Save figure as .pdf ------------------------ #
######################################################################

ggsave("Figure_S1.pdf", plot = linear_models, device = 'pdf', path = 'results/figures', width = 8, height = 2.5)

ggsave("Figure_S2.pdf", plot = non_linear_models, device = 'pdf', path = 'results/figures', width = 7, height = 6)
