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
#detach("package:randomForest", unload=TRUE) 
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
# Read in the cv AUROC results of trained model of 100 data-splits
######################################################################
all_files <- list.files(path= 'data/process', pattern='combined_all_hp.*', full.names = TRUE) 

fig_data <- map_df(all_files[2:4], read_csv)

######################################################################
# Plot the mean cvAUC values for hyper parameters tested #
######################################################################

linear_plots <- fig_data %>%
  group_by(model, cost) %>%
  summarize(mean_AUC=mean(ROC), sd_AUC=sd(ROC)) %>%
  ungroup() %>%
  ggplot(aes(x=cost, y=mean_AUC, color=model)) +
  geom_hline(yintercept=0.5, linetype="dashed") +
  geom_line(size=1.5) +
  geom_point(size=5) +
  geom_errorbar(aes(ymin=mean_AUC-sd_AUC, ymax=mean_AUC+sd_AUC), width=.001) +
  scale_x_log10(limits=c(min=1e-4,max=1),
                breaks=c(1e-4, 1e-3, 1e-2, 1e-1, 1),
                name="regularization penalty (C)",
                labels=trans_format('log10',math_format(10^.x))) +
  scale_y_continuous(name="mean cvAUROC",
                     limits=c(0.4,1.0),
                     breaks=seq(0.4,1,0.1)) +
  scale_color_manual(name=NULL,
                     labels=c(expression(paste(L[1], "-regularized linear kernel SVM")),
                              expression(paste(L[2], "-regularized linear kernel SVM")),
                              expression(paste(L[2], "-regularized logistic regression"))),
                     breaks=c("L1_Linear_SVM", "L2_Linear_SVM", "L2_Logistic_Regression"),
                     values=c("#00AFBB", "#E7B800", "#FC4E07"))+
  theme_bw() +
  theme(legend.background = element_rect(linetype="solid", color="black", size=0.5),
        legend.box.margin=margin(c(12,12,12, 12)),
        legend.text=element_text(size=18),
        legend.title=element_blank(),
        legend.position=c(0.65, 0.85),
        panel.grid.major = element_blank(),
        panel.grid.minor = element_blank(),
        panel.background = element_blank(),
        text = element_text(size = 18),
        axis.text.x=element_text(size = 20, colour='black'),
        axis.text.y=element_text(size = 20, colour='black'),
        axis.title.y=element_text(size = 24),
        axis.title.x=element_text(size = 24),
        panel.border = element_rect(linetype="solid", colour = "black", fill=NA, size=1.5))


######################################################################
#-----------------------Save figure as .pdf ------------------------ #
######################################################################

ggsave("Figure_S1.png", plot = linear_plots, device = 'png', path = 'submission', width = 7, height = 5)

