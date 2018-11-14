# Author: Begum Topcuoglu
# Date: 2018-11-14
#
######################################################################
# 1. This script loads the *_parameters.tsv files generated from python code main.py for all the models. The .tsv files have mean AUC values of all the hyper-parameters tested on the training data for the 100 times that outer-validation runs. 
# 2. It also plots the hyper-parameter range and performance during cross validation. 
######################################################################

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
#------------ Load .tsv data generated in Python -----------------#
######################################################################

# Read in hyper-parameter AUCs generated from L2 logistic regression model for all samples:
logit <- read.delim('data/process/L2_Logistic_Regression_parameters.tsv', header=T, sep='\t')%>%
  select(-X) %>% 
  mutate(meanAUC=rowMeans(.[, 2:101])) %>% 
  select(C, meanAUC) 

logit_plot <- ggplot(logit,aes(x=C,y=meanAUC)) + 
  geom_line() + 
  geom_point() +
  scale_y_continuous(name="Logistic Regression cvAUC", 
                     limits = c(0.50, 0.80), 
                     breaks = seq(0.5, 0.8, 0.05)) + 
  scale_x_continuous(name="C (penalty)") +
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

# Read in AUCs table generated from l1 SVM linear kernel for all samples
#       Carcinomas + Adenomas are 1 and Normal is 0 for binary.
#       FIT is a feature
#       The scaler is Standard scaler
l1svm <- read.delim('data/process/L1_SVM_Linear_Kernel_parameters.tsv', header=T, sep='\t')%>%
  select(-X) %>% 
  mutate(meanAUC=rowMeans(.[, 2:101])) %>% 
  select(C, meanAUC) 

l1svm_plot <- ggplot(l1svm,aes(x=C,y=meanAUC)) + 
  geom_line() + 
  geom_point() +
  scale_y_continuous(name="L1 Linear SVM cvAUC", 
                     limits = c(0.50, 0.80), 
                     breaks = seq(0.5, 0.8, 0.05)) + 
  scale_x_continuous(name="C (penalty)") +
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


# Read in AUCs table generated from l2 SVM linear kernel for all samples
#       Carcinomas + Adenomas are 1 and Normal is 0 for binary.
#       FIT is a feature
#       The scaler is Standard scaler
l2svm <- read.delim('data/process/L2_SVM_Linear_Kernel_parameters.tsv', header=T, sep='\t')%>%
  select(-X) %>% 
  mutate(meanAUC=rowMeans(.[, 2:101])) %>% 
  select(C, meanAUC) 

l2svm_plot <- ggplot(l2svm,aes(x=C,y=meanAUC)) + 
  geom_line() + 
  geom_point() +
  scale_y_continuous(name="L2 Linear SVM cvAUC", 
                     limits = c(0.50, 0.80), 
                     breaks = seq(0.5, 0.8, 0.05)) + 
  scale_x_continuous(name="C (penalty)") +
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

plot_grid(logit_plot, l1svm_plot, l2svm_plot, labels = c("A", "B", "C"))



# Read in AUCs table generated from  SVM RBF kernel for all samples
#       Carcinomas + Adenomas are 1 and Normal is 0 for binary.
#       FIT is a feature
#       The scaler is Standard scaler
svmRBF <- read.delim('data/process/SVM_RBF_parameters.tsv', header=T, sep='\t')%>%
  select(-X) %>% 
  mutate(meanAUC=rowMeans(.[, 2:101])) %>% 
  select(gamma, C, meanAUC) 

ggplot(l2svm, aes(x=C,y=meanAUC)) + geom_line()

# Read in AUCs table generated from xgboost for all samples
#       Carcinomas + Adenomas are 1 and Normal is 0 for binary.
#       FIT is a feature
#       The scaler is MinMax (0-1) scaler
xgboost <- read.delim('../../data/process/XGBoost.tsv', header=T, sep='\t') %>%
  select(level_1, AUC) %>%
  rename(Performance = level_1) %>%
  mutate(model="XGBoost")

# Read in AUCs table generated from random forest for all samples
#       Carcinomas + Adenomas are 1 and Normal is 0 for binary.
#       FIT is a feature
#       The scaler is MinMax (0-1) scaler
rf <- read.delim('../../data/process/Random_Forest.tsv', header=T, sep='\t') %>%
  select(level_1, AUC) %>%
  rename(Performance = level_1) %>%
  mutate(model="Random Forest ")

# Read in AUCs table generated from decision tree for all samples
#       Carcinomas + Adenomas are 1 and Normal is 0 for binary.
#       FIT is a feature
#       The scaler is MinMax (0-1) scaler
dt <- read.delim('../../data/process/Decision_Tree.tsv', header=T, sep='\t') %>%
  select(level_1, AUC) %>%
  rename(Performance = level_1) %>%
  mutate(model="Decision Tree")

######################################################################
#------------ Put all the loaded AUC tables together -----------------#
######################################################################

all <- bind_rows(logit, l1svm, l2svm, svmRBF, xgboost, rf, dt) %>%
  group_by(model)
