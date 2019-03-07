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
deps = c("cowplot", "ggplot2","knitr","rmarkdown","vegan","gtools", "tidyverse");
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

all_files <- list.files(path= 'data/process', pattern='combined_all.*', full.names = TRUE)

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
  theme(legend.text=element_text(size=18),
        legend.title=element_text(size=22),
        panel.grid.major = element_blank(),
        panel.grid.minor = element_blank(),
        panel.background = element_blank(),
        text = element_text(size = 12),
        axis.text.x=element_text(size = 12, colour='black'),
        axis.text.y=element_text(size = 12, colour='black'),
        axis.title.y=element_text(size = 13),
        axis.title.x=element_text(size = 13))
  return(plot)
}

# Start plotting models with one hyper-parameter individually
l1svm <- l1svm_all %>%
  group_by(cost) %>%
  summarise(mean_Acc = mean(Accuracy), sd_Acc = sd(Accuracy))

l1svm_plot <- base_plot(l1svm, l1svm$cost, l1svm$mean_Acc) +
  scale_x_continuous(name="C (penalty)") +
  scale_y_continuous(name="L1 SVM with linear kernel mean cvAUC",
                     limits = c(0.30, 1),
                     breaks = seq(0.3, 1, 0.05)) +
  geom_errorbar(aes(ymin=mean_Acc-sd_Acc, ymax=mean_Acc+sd_Acc), width=.001)


l2svm <- l2svm_all %>%
  group_by(cost) %>%
  summarise(mean_Acc = mean(Accuracy), sd_Acc = sd(Accuracy))

l2svm_plot <- base_plot(l2svm, l2svm$cost, l2svm$mean_Acc) +
  scale_x_continuous(name="C (penalty)", 
                     breaks= c(0.0005, 0.001, 0.005, 0.01, 0.05, 0.1, 0.5, 1)) +
  scale_y_continuous(name="L2 SVM with linear kernel mean cvAUC",
                     limits = c(0.30, 1),
                     breaks = seq(0.3, 1, 0.05)) +
  geom_errorbar(aes(ymin=mean_Acc-sd_Acc, ymax=mean_Acc+sd_Acc))

logit_plot <- logit_all %>%
  group_by(cost, loss, epsilon) %>%
  summarise(mean_AUC = mean(ROC), sd_AUC = sd(ROC)) %>%
  ggplot(aes(x=cost,y=mean_AUC)) +
  geom_line() +
  geom_point() +
  scale_x_continuous(name="C (penalty)") +
  scale_y_continuous(name="L2 logistic regression mean cvAUC",
                     limits = c(0.30, 1),
                     breaks = seq(0.3, 1, 0.05)) +
  geom_errorbar(aes(ymin=mean_AUC-sd_AUC, ymax=mean_AUC+sd_AUC), width=.001) +
  theme_bw() +
  theme(legend.text=element_text(size=18),
        legend.title=element_text(size=22),
        panel.grid.major = element_blank(),
        panel.grid.minor = element_blank(),
        panel.background = element_blank(),
        text = element_text(size = 12),
        axis.text.x=element_text(size = 12, colour='black'),
        axis.text.y=element_text(size = 12, colour='black'),
        axis.title.y=element_text(size = 13),
        axis.title.x=element_text(size = 13))

dt <- dt_all %>%
  group_by(maxdepth) %>%
  summarise(mean_AUC = mean(ROC), sd_AUC = sd(ROC))

dt_plot <- base_plot(dt, dt$maxdepth, dt$mean_AUC) +
scale_x_continuous(name="max depth") +
  scale_y_continuous(name="Decision tree mean cvAUC",
                     limits = c(0.30, 1),
                     breaks = seq(0.3, 1, 0.05)) +
  geom_errorbar(aes(ymin=mean_AUC-sd_AUC, ymax=mean_AUC+sd_AUC), width=.001)

rf <- rf_all %>%
  group_by(mtry) %>%
  summarise(mean_AUC = mean(ROC), sd_AUC = sd(ROC))

rf_plot <-  base_plot(rf, rf$mtry, rf$mean_AUC) +
scale_x_continuous(name="mtry",
                   breaks=seq(0, 1500, 250), limits = c(0, 1500)) +
  scale_y_continuous(name="Random forest mean cvAUC",
                     limits = c(0.30, 1),
                     breaks = seq(0.3, 1, 0.05)) +
  geom_errorbar(aes(ymin=mean_AUC-sd_AUC, ymax=mean_AUC+sd_AUC), width=1)

# Start plotting models with 2 hyper-parameters individually

rbf_data<- rbf_all %>%
  group_by(sigma, C) %>%
  summarise(mean_AUC = mean(ROC), sd_AUC = sd(ROC)) %>% 
  filter(sigma!=1)

library(scales)
rbf_data$sigma <- scientific(rbf_data$sigma)
rbf_data$sigma <- as.character(rbf_data$sigma)
  
rbf_plot <- ggplot(rbf_data, aes(x=C, y=mean_AUC, color=sigma)) +
  geom_line() +
  geom_point() + 
  scale_x_log10(breaks = c(0.0000001, 0.000001, 0.00001, 0.0001, 0.001, 0.01, 0.1, 1, 10)) +
  scale_color_brewer(palette="Dark2") +
  scale_y_continuous(name="SVM with radial bias kernel mean cvAUC",
                     limits = c(0.50, 0.75))

xgboost_data <- xgboost_all %>%
  group_by(eta, subsample) %>%
  summarise(mean_AUC = mean(ROC), sd_AUC = sd(ROC)) 


xgboost_data$subsample <- as.character(xgboost_data$subsample)
xgboost_plot <- ggplot(xgboost_data, aes(x=eta, y=mean_AUC, color=subsample)) +
  geom_line() +
  geom_point()+
  scale_x_continuous(breaks = c(0.001, 0.005, 0.01, 0.015, 0.02)) +
  scale_y_continuous(name="XGBoost mean cvAUC",
                     limits = c(0.80, 0.82),
                     breaks = seq(0.80, 0.82, 0.005)) +
  scale_color_brewer(palette="Dark2")

  
  



all <- plot_grid(logit_plot, l1svm_plot, l2svm_plot, rbf_plot, rf_plot, dt_plot, xgboost_plot, labels = c("A", "B", "C", "D", "E", "F", "G"))

######################################################################
#-----------------------Save figure as .pdf ------------------------ #
######################################################################

ggsave("Figure_2.pdf", plot = all, device = 'pdf', path = 'results/figures', width = 20, height = 15)
