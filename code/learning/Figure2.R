# Author: Begum Topcuoglu
# Date: 2018-02-12
#
######################################################################
# This script plots Figure 1:
#   1. cvAUC (means of 100 repeats for the best hp) of 100 datasplits
#   2. testAUC of 100 datasplits
######################################################################

######################################################################
# Load in needed functions and libraries
source('code/learning/functions.R')
# detach("package:randomForest", unload=TRUE) to run
######################################################################


######################################################################
# Load .csv data generated with modeling pipeline
######################################################################

# Read in the cvAUCs, testAUCs for 100 splits.
best_files <- list.files(path= 'data/process', pattern='combined_best.*', full.names = TRUE)

best_performance <- map_df(best_files, read_files) %>% 
  melt_data()

######################################################################
#Plot the AUC values for cross validation and testing for each model #
######################################################################
# Permutation test of means from two groups 
all <- map_df(best_files, read_files) 
set.seed(101)
#resampling method 
rf_logit <- perm_p_value(all, "L2_Logistic_Regression","Random_Forest") # different p=0.00049995
perm_p_value(all,"XGBoost", "L2_Logistic_Regression") # not different
perm_p_value(all, "XGBoost", "L2_Linear_SVM") # not different
perm_p_value(all, "RBF_SVM", "L2_Linear_SVM") # not different
perm_p_value(all, "L1_Linear_SVM", "RBF_SVM") # different p = 0.00069993
perm_p_value(all, "Decision_Tree", "RBF_SVM") # different p = 9.999e-05
perm_p_value(all, "L1_Linear_SVM", "Decision_Tree") # different p = 9.999e-05

performance <- ggplot(best_performance, aes(x = fct_reorder(model, AUC), y = AUC, fill = Performance)) +
  geom_boxplot(alpha=0.5, fatten = 2) +
  geom_hline(yintercept = 0.5, linetype="dashed") +
  scale_fill_manual(values=c("blue4", "springgreen4"), 
                    guide = guide_legend(reverse = TRUE)) +
  coord_flip() +
  scale_y_continuous(name = "AUROC",
                     breaks = seq(0.4, 1, 0.1),
                     limits=c(0.4, 1),
                     expand=c(0,0)) +
  scale_x_discrete(name = "",
                   labels=c("Decision tree",
                           expression(paste(L[1], "-regularized linear SVM")),
                            "SVM with radial basis kernel",
                            expression(paste(L[2], "-regularized linear SVM")),
                            "XGBoost",
                            expression(paste(L[2], "-regularized logistic regression")),
                            "Random forest")) + 
  geom_signif(comparisons=list(c("RBF_SVM", "Random_Forest")), annotations="",
              y_position = 0.83, tip_length = 0.02, vjust=0.3) +
  annotate("text", x= 5, y= 0.86, size=4, label = paste0("N.S.")) + 
  theme_bw() +
  theme(plot.margin=unit(c(0,1.1,0,0),"cm"),
        legend.justification=c(1,0),
        legend.position=c(1,0),
        #legend.position="bottom",
        legend.title = element_blank(),
        legend.background = element_rect(linetype="solid", color="black", size=0.5),
        legend.box.margin=margin(c(10,10,10, 10)),
        legend.text=element_text(size=10),
        #legend.title=element_text(size=22),
        panel.grid.major.y = element_blank(),
        panel.grid.major.x = element_line( size=0.6),
        panel.grid.minor = element_blank(),
        panel.background = element_blank(),
        text = element_text(size = 12),
        axis.text.x=element_text(size = 10, colour='black'),
        axis.text.y=element_text(size = 10, colour='black'),
        axis.title.y=element_text(size = 10),
        axis.title.x=element_text(size = 10),
        panel.border = element_rect(linetype="solid", colour = "black", fill=NA, size=1.5))

######################################################################
#-----------------------Save figure as .pdf ------------------------ #
######################################################################

ggsave("Figure_2.tiff", compression = "lzw", plot = performance, device = 'tiff', path = 'submission', units="in",width=6.875, height=4.5)

