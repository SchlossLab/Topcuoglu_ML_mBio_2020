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
logit <- read.delim('data/process/L2_Logistic_Regression_aucs_hps_R.tsv', header=T, sep='\t') %>% 
  melt_data %>% 
  mutate(model='L2 Logistic Regression')  


l2svm <- read.delim('data/process/L2_Linear_SVM_aucs_hps_R.tsv', header=T, sep='\t')%>% 
  melt_data%>% 
  mutate(model='L2 Linear SVM')

rbf <- read.delim('data/process/SVM_RBF_aucs_hps_R.tsv', header=T, sep='\t')%>% 
  melt_data%>% 
  mutate(model='RBF SVM') 

xgboost <- read.delim('data/process/xgboost_aucs_hps_R.tsv', header=T, sep='\t')%>% 
  melt_data%>% 
  mutate(model='XGBoost') 

dt <-  read.delim('data/process/decision_tree_aucs_hps_R.tsv', header=T, sep='\t')%>% 
  melt_data%>% 
  mutate(model='Decision Tree')

all <- bind_rows(logit, l2svm, rbf, dt, xgboost) %>%
  group_by(model)

######################################################################
#Plot the AUC values for cross validation and testing for each model #
######################################################################


ggplot(all, aes(x = fct_reorder(model, AUC, fun = median, .asc =TRUE), y = AUC, fill = Performance)) +
  geom_boxplot(alpha=0.7) +
  scale_y_continuous(name = "AUC",
                     breaks = seq(0.5, 1, 0.02),
                     limits=c(0.5, 1), expand=c(0,0)) +
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
        axis.title.x=element_text(size = 20)) +
  scale_fill_brewer(palette = "Paired")


######################################################################
#-----------------------Save figure as .pdf ------------------------ #
######################################################################

ggsave("AUC_comparison_R.pdf", plot = last_plot(), device = 'pdf', path = 'results/figures', width = 10, height = 10)


######################################################################
#Plot the mean AUC values for hyper parameters tested #
######################################################################

logit_HP_plot <- plot_parameter_linear(logit) +  scale_y_continuous(name="Logistic Regression mean AUC",
                                                    limits = c(0.50, 1),
                                                    breaks = seq(0.5, 1, 0.05))

l2svm_HP_plot <- plot_parameter_linear(l2svm) +   scale_y_continuous(name="L2 Support Vector Machine mean AUC",
                                                                     limits = c(0.50, 1),
                                                                     breaks = seq(0.5, 1, 0.05))

l2svm_HP_plot <- plot_parameter_linear(l2svm) +   scale_y_continuous(name="L2 Support Vector Machine mean AUC",
                                                                     limits = c(0.50, 1),
                                                                     breaks = seq(0.5, 1, 0.05))


