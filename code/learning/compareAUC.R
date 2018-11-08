# Author: Begum Topcuoglu
# Date: 2018-11-06
#
######################################################################
# 1. This script loads the .tsv files generated from python code main.py for all the models. The .tsv files have AUC values for cross validation (for each repeat and n-fold) and for testing for each iteration over different splits of the training-testing data. 
# 2. This script then binds all the models together and makes a boxplot for AUC for all the models. 
######################################################################


######################################################################
#----------------- Read in necessary libraries -------------------#
######################################################################
deps = c("ggplot2","knitr","rmarkdown","vegan","gtools", "tidyverse");
for (dep in deps){
  if (dep %in% installed.packages()[,"Package"] == FALSE){
    install.packages(as.character(dep), quiet=TRUE);
  }
  library(dep, verbose=FALSE, character.only=TRUE)
}

######################################################################
#------------ Load .tsv data generated in Python -----------------#
######################################################################

# Read in AUCs table generated from L2 logistic regression model for all samples:
#       Carcinomas + Adenomas are 1 and Normal is 0 for binary.
#       FIT is a feature
#       The scaler is MinMax scaler
logit <- read.delim('data/process/L2_Logistic_Regression.tsv', header=T, sep='\t') %>%
  select(level_1, AUC) %>% 
  rename(Performance = level_1) %>% 
  mutate(model="L2-Logistic Regression")

# Read in AUCs table generated from l1 SVM linear kernel for all samples
#       Carcinomas + Adenomas are 1 and Normal is 0 for binary.
#       FIT is a feature
#       The scaler is Standard scaler
l1svm <- read.delim('data/process/L1_SVM_Linear_Kernel.tsv', header=T, sep='\t') %>%
  select(level_1, AUC) %>% 
  rename(Performance = level_1) %>% 
  mutate(model="L1-SVM Linear") 

# Read in AUCs table generated from l2 SVM linear kernel for all samples
#       Carcinomas + Adenomas are 1 and Normal is 0 for binary.
#       FIT is a feature
#       The scaler is Standard scaler
l2svm <- read.delim('data/process/L2_SVM_Linear_Kernel.tsv', header=T, sep='\t') %>%
  select(level_1, AUC) %>% 
  rename(Performance = level_1) %>% 
  mutate(model="L2-SVM Linear")

# Read in AUCs table generated from  SVM RBF kernel for all samples
#       Carcinomas + Adenomas are 1 and Normal is 0 for binary.
#       FIT is a feature
#       The scaler is Standard scaler
svmRBF <- read.delim('data/process/SVM_RBF.tsv', header=T, sep='\t') %>%
  select(level_1, AUC) %>% 
  rename(Performance = level_1) %>% 
  mutate(model="SVM RBF")

# Read in AUCs table generated from xgboost for all samples
#       Carcinomas + Adenomas are 1 and Normal is 0 for binary.
#       FIT is a feature
#       The scaler is MinMax (0-1) scaler
xgboost <- read.delim('data/process/XGBoost.tsv', header=T, sep='\t') %>%
  select(level_1, AUC) %>% 
  rename(Performance = level_1) %>% 
  mutate(model="XGBoost")

# Read in AUCs table generated from random forest for all samples
#       Carcinomas + Adenomas are 1 and Normal is 0 for binary.
#       FIT is a feature
#       The scaler is MinMax (0-1) scaler
rf <- read.delim('data/process/Random_Forest.tsv', header=T, sep='\t') %>%
  select(level_1, AUC) %>% 
  rename(Performance = level_1) %>% 
  mutate(model="Random Forest ")

# Read in AUCs table generated from decision tree for all samples
#       Carcinomas + Adenomas are 1 and Normal is 0 for binary.
#       FIT is a feature
#       The scaler is MinMax (0-1) scaler
dt <- read.delim('data/process/Decision_Tree.tsv', header=T, sep='\t') %>%
  select(level_1, AUC) %>% 
  rename(Performance = level_1) %>% 
  mutate(model="Decision Tree")

######################################################################
#------------ Put all the loaded AUC tables together -----------------#
######################################################################

all <- bind_rows(logit, l1svm, l2svm, svmRBF, xgboost, rf, dt) %>% 
  group_by(model)

######################################################################
#Plot the AUC values for cross validation and testing for each model #
######################################################################

ggplot(all, aes(x = fct_reorder(model, AUC, fun = median, .asc =TRUE), y = AUC, fill = Performance)) +
  geom_boxplot(alpha=0.7) +
  scale_y_continuous(name = "AUC",
                     breaks = seq(0.3, 1, 0.05),
                     limits=c(0.3, 1), expand=c(0,0)) +
  scale_x_discrete(name = "") +
  theme_bw() +
  theme(legend.text=element_text(size=18),
        legend.title=element_text(size=22),
        panel.grid.major = element_blank(), 
        panel.grid.minor = element_blank(),
        panel.background = element_blank(),
        text = element_text(size = 12),
        axis.text.x=element_text(size = 12, colour='black'),
        axis.text.y=element_text(size = 12, colour='black'), 
        axis.title.y=element_text(size = 20), 
        axis.title.x=element_text(size = 20)) +
  scale_fill_brewer(palette = "Paired") +
  geom_hline(yintercept = 0.5, linetype="dashed") +
  annotate(geom="segment", y=seq(0.3,1,0.01), yend = seq(0.3,1,0.01),
           x=0, xend=0.03)

######################################################################
#-----------------------Save figure as .pdf ------------------------ #
######################################################################

ggsave("AUC_comparison.pdf", plot = last_plot(), device = 'pdf', path = 'results/figures', width = 15, height = 10)

  
