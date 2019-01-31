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

filenames <- list.files(path= 'data/process/', pattern='.*parameters.tsv')

# Read in hyper-parameter AUCs generated from L2 logistic regression model for all samples:
logit <- read.delim('data/process/L2_Logistic_Regression_parameters.tsv', header=T, sep='\t')%>%
  select(-X) %>%
  mutate(meanAUC=rowMeans(.[, 2:101])) %>% # Take the mean of the rows for each C
  select(C, meanAUC)

logit_plot <- ggplot(logit,aes(x=C,y=meanAUC)) +
  geom_line() +
  geom_point() +
  scale_y_continuous(name="Logistic Regression cvAUC",
                     limits = c(0.50, 1),
                     breaks = seq(0.5, 1, 0.05)) +
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
                     limits = c(0.50, 1),
                     breaks = seq(0.5, 1, 0.05)) +
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
                     limits = c(0.50, 1),
                     breaks = seq(0.5, 1, 0.05)) +
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

# Read in hyper-parameter AUCs generated from SVM RBF model for all samples:

svmRBF <- read.delim('data/process/SVM_RBF_parameters.tsv', header=T, sep='\t')%>%
  select(-X) %>%
  mutate(meanAUC=rowMeans(.[, 3:102])) %>%  # Take the mean of the rows for each C
  select(C, gamma, meanAUC) %>% 
  group_by(gamma) 

svmRBF_plot_C_sigma <- ggplot(svmRBF,aes(x=C,y=meanAUC))+
  geom_point() +
  facet_grid(~gamma, labeller = labeller()) +
  geom_line() +
  scale_y_continuous(name="RBF SVM cvAUC",
                     limits = c(.70, .71),
                     breaks = seq(.70, .71, 0.001)) +
  scale_x_log10(labels = scales::trans_format("log10", scales::math_format(10^.x))) +
  theme_bw() +
  theme(panel.spacing = unit(1, "lines"),
        legend.text=element_text(size=18),
        legend.title=element_text(size=22),
        panel.grid.major = element_blank(),
        panel.grid.minor = element_blank(),
        panel.background = element_blank(),
        text = element_text(size = 12),
        axis.text.x=element_text(size = 12, colour='black'),
        axis.text.y=element_text(size = 12, colour='black'),
        axis.title.y=element_text(size = 13),
        axis.title.x=element_text(size = 13))


rf <- read.delim('data/process/Random_Forest_parameters.tsv', header=T, sep='\t')%>%
  select(-X) %>%
  mutate(meanAUC=rowMeans(.[, 3:102])) %>% # Take the mean of the rows for each C
  select(max_features, meanAUC)
  

rf_plot_max <- ggplot(rf,aes(x=max_features,y=meanAUC)) +
  geom_line() +
  geom_point() +
  scale_y_continuous(name="Random Forest cvAUC",
                     limits = c(0.50, 1),
                     breaks = seq(0.5, 1, 0.05)) +
  scale_x_continuous(name="max features") +
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

dt <- read.delim('data/process/Decision_Tree_parameters.tsv', header=T, sep='\t')%>%
  select(-X) %>%
  mutate(meanAUC=rowMeans(.[, 3:102])) %>% # Take the mean of the rows for each C
  select(max_depth, min_samples_split, meanAUC) %>% 
  group_by(max_depth)

dt_plot_max <- ggplot(dt,aes(x=min_samples_split,y=meanAUC)) +
  geom_point() +
  facet_grid(~max_depth) +
  geom_line() +
  scale_y_continuous(name="Decision Tree cvAUC",
                     limits = c(0.65, 0.75),
                     breaks = seq(0.65, 0.75, 0.01)) +
  scale_x_continuous(name="max depth") +
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


all <- plot_grid(logit_plot, l1svm_plot, l2svm_plot, svmRBF_plot_C_sigma, rf_plot_max, dt_plot_max, labels = c("A", "B", "C", "D", "E", "F"))





######################################################################
#-----------------------Save figure as .pdf ------------------------ #
######################################################################

ggsave("HP_comparison_python.pdf", plot = all, device = 'pdf', path = 'results/figures', width = 20, height = 15)

