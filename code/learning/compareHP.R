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
  mutate(meanAUC=rowMeans(.[, 2:101])) %>% # Take the mean of the rows for each C
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

# Read in hyper-parameter AUCs generated from L1 Linear SVM model for all samples:
l1svm <- read.delim('data/process/L1_SVM_Linear_Kernel_parameters.tsv', header=T, sep='\t')%>%
  select(-X) %>%
  mutate(meanAUC=rowMeans(.[, 2:101])) %>%  # Take the mean of the rows for each C
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


# Read in hyper-parameter AUCs generated from L2 Linear SVM model for all samples:

l2svm <- read.delim('data/process/L2_SVM_Linear_Kernel_parameters.tsv', header=T, sep='\t')%>%
  select(-X) %>%
  mutate(meanAUC=rowMeans(.[, 2:101])) %>%  # Take the mean of the rows for each C
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





######################################################################
#-----------------------Save figure as .pdf ------------------------ #
######################################################################

ggsave("HP_comparison.pdf", plot = last_plot(), device = 'pdf', path = '/Users/btopcuoglu/Documents/DeepLearning/results/figures', width = 15, height = 10)
