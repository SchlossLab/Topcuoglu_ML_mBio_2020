# Author: Begum Topcuoglu
# Date: 2018-12-05
#
######################################################################
# This script plots the cv and test AUC values as a boxplot for all models
######################################################################


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

######################################################################
# Load .tsv data generated with modeling pipeline for Logistic Regression
######################################################################

# Read in the cvAUCs, testAUCs and hyper-parameters from L2 logistic regression model for 100 splits.
best_files <- list.files(path= 'data/process', pattern='combined_best.*', full.names = TRUE)

read_files <- function(filenames){
  for(file in filenames){
    # Read the files generated in main.R 
    # These files have cvAUCs and testAUCs for 100 data-splits
    data <- read.delim(file, header=T, sep=',')
  }
  return(data)
}

logit <- read_files(best_files[4])
l2svm <- read_files(best_files[3])
l1svm <- read_files(best_files[2])
rbf <- read_files(best_files[6])
rf <- read_files(best_files[5])
dt <- read_files(best_files[1])
xgboost <- read_files(best_files[7])



best_performance <- bind_rows(logit, l2svm, rbf, dt, xgboost, rf, l1svm)%>%
  melt_data()

######################################################################
#Plot the AUC values for cross validation and testing for each model #
######################################################################


performance <- ggplot(best_performance, aes(x = fct_reorder(model, AUC, fun = median, .asc =TRUE), y = AUC, fill = Performance)) +
  geom_boxplot(alpha=0.7) +
  scale_y_continuous(name = "AUC",
                     breaks = seq(0.5, 1, 0.02),
                     limits=c(0.5, 1), 
                     expand=c(0,0)) +
  scale_x_discrete(name = "") +
  theme_bw() +
  theme(legend.justification=c(0,1), 
        legend.position=c(0,1),
        legend.box.margin=margin(c(10,10,10,10)),
        legend.text=element_text(size=18),
        legend.title=element_text(size=22),
        panel.grid.major = element_blank(),
        panel.grid.minor = element_blank(),
        panel.background = element_blank(),
        text = element_text(size = 12),
        axis.text.x=element_text(size = 12, colour='black'),
        axis.text.y=element_text(size = 12, colour='black'),
        axis.title.y=element_text(size = 20),
        axis.title.x=element_text(size = 20)) 


######################################################################
#-----------------------Save figure as .pdf ------------------------ #
######################################################################

ggsave("AUC_comparison_R.pdf", plot = performance, device = 'pdf', path = 'results/figures', width = 15, height = 10)


######################################################################
#Plot the mean AUC values for hyper parameters tested #
######################################################################

all_files <- list.files(path= 'data/process', pattern='combined_all.*', full.names = TRUE)


logit_all <- read_files(all_files[4])
l2svm_all <- read_files(all_files[3])
l1svm_all <- read_files(all_files[2])
rbf_all <- read_files(all_files[6])
rf_all <- read_files(all_files[5])
dt_all <- read_files(all_files[1])
xgboost_all <- read_files(all_files[7])

logit_all %>% 
  group_by(cost, loss, epsilon) %>% 
  summarise(mean_AUC = mean(ROC), sd_AUC = sd(ROC)) %>% 
  ggplot(aes(x=cost,y=mean_AUC)) +
  geom_line() +
  geom_point() +
  scale_x_continuous(name="C (penalty)") +
  scale_y_continuous(name="L2 Logistic Regression mean AUC",
                     limits = c(0.50, 1),
                     breaks = seq(0.5, 1, 0.05)) +
  geom_errorbar(aes(ymin=mean_AUC-sd_AUC, ymax=mean_AUC+sd_AUC), width=.01) +
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


l2svm_all %>% 
  group_by(C) %>% 
  summarise(mean_AUC = mean(ROC), sd_AUC = sd(ROC)) %>% 
  ggplot(aes(x=C,y=mean_AUC)) +
  geom_line() +
  geom_point() +
  scale_x_continuous(name="C (penalty)") +
  scale_y_continuous(name="L2 Linear Kernel SVM mean AUC",
                     limits = c(0.50, 1),
                     breaks = seq(0.5, 1, 0.05)) +
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

l1svm_all %>% 
  group_by(cost) %>% 
  summarise(mean_sens = mean(Sens), mean_spec = mean(Spec)) %>% 
  ggplot(aes(x=C,y=mean_AUC)) +
  geom_line() +
  geom_point() +
  scale_x_continuous(name="C (penalty)") +
  scale_y_continuous(name="L2 Linear Kernel SVM mean AUC",
                     limits = c(0.50, 1),
                     breaks = seq(0.5, 1, 0.05)) +
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


rbf_all %>% 
  group_by(sigma, C) %>% 
  summarise(mean_AUC = mean(ROC), sd_AUC = sd(ROC)) %>% 
  group_by(C) %>% 
  ggplot(aes(x=sigma,y=mean_AUC)) +
  facet_grid(~C) + 
  geom_line() +
  geom_point() +
  scale_x_log10(labels = scales::trans_format("log10", scales::math_format(10^.x)),
                breaks= c(1e-08, 1e-07,1e-06, 1e-05)) +
  theme_bw() +
  theme(legend.position="none",
        panel.grid.major = element_blank(),
        panel.grid.minor = element_blank(),
        panel.background = element_blank(),
        text = element_text(size = 12),
        axis.text.x=element_text(size = 11, colour='black'),
        axis.text.y=element_text(size = 11, colour='black'),
        axis.title.y=element_text(size = 13),
        axis.title.x=element_text(size = 13)) +
  scale_y_continuous(name="SVM Support Vector Machine mean AUC",
                     limits = c(0.50, 1),
                     breaks = seq(0.5, 1, 0.05))

rf_all %>% 
  group_by(mtry) %>% 
  summarise(mean_AUC = mean(ROC), sd_AUC = sd(ROC)) %>% 
  ggplot(aes(x=mtry,y=mean_AUC)) +
  geom_line() +
  geom_point() +
  scale_x_continuous(name="mtry", 
                     breaks=seq(0, 1500, 250), limits = c(0, 1500)) +
  scale_y_continuous(name="Random Forest mean cvAUC",
                     limits = c(0.50, 1),
                     breaks = seq(0.5, 1, 0.05)) +
  geom_errorbar(aes(ymin=mean_AUC-sd_AUC, ymax=mean_AUC+sd_AUC), width=2) +
  theme_bw() +
  theme(legend.text=element_text(size=18),
        legend.title=element_text(size=22),
        panel.grid.major = element_blank(),
        panel.grid.minor = element_blank(),
        panel.background = element_blank(),
        text = element_text(size = 12),
        axis.text.x=element_text(size = 10, colour='black'),
        axis.text.y=element_text(size = 12, colour='black'),
        axis.title.y=element_text(size = 13),
        axis.title.x=element_text(size = 13))


dt_all %>% 
  group_by(maxdepth) %>% 
  summarise(mean_AUC = mean(ROC), sd_AUC = sd(ROC)) %>% 
  ggplot(aes(x=maxdepth,y=mean_AUC)) +
  geom_line() +
  geom_point() +
  scale_x_continuous(name="max depth") +
  scale_y_continuous(name="Random Forest mean AUC",
                     limits = c(0.50, 1),
                     breaks = seq(0.5, 1, 0.05)) +
  geom_errorbar(aes(ymin=mean_AUC-sd_AUC, ymax=mean_AUC+sd_AUC), width=0.2) +
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

xgboost_all %>% 
  group_by(eta, nrounds, subsample) %>% 
  summarise(mean_AUC = mean(ROC), sd_AUC = sd(ROC)) %>% 
  group_by(min_child_weight) %>% 
  ggplot(aes(x=subsample,y=mean_AUC)) +
  facet_grid(~min_child_weight) + 
  geom_line() +
  geom_point() +
  scale_x_continuous(name="subsample") +
  scale_y_continuous(name="XGBoost mean AUC",
                     limits = c(0.50, 1),
                     breaks = seq(0.5, 1, 0.05)) +
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


