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
perm_p_value <- function(data, model_name_1, model_name_2){
  test_subsample <- all %>% 
    select(-cv_aucs) %>% 
    filter(model == model_name_1 | model == model_name_2)
  library(mosaic)
  obs <- abs(diff(mosaic::mean(test_aucs ~ model, data = test_subsample)))
  auc.null <- do(10000) * diff(mosaic::mean(test_aucs ~ shuffle(model), data = test_subsample))
  n <- length(auc.null[,1])
  # r = #replications at least as extreme as observed effect
  r <- sum(abs(auc.null[,1]) >= obs)  
  # compute Monte Carlo p-value with correction (Davison & Hinkley, 1997)
  p.value=(r+1)/(n+1)
  plot <- ggplot(auc.null, aes(x=auc.null[,1], color=auc.null[,1]>=obs)) + geom_histogram(fill="white", alpha=0.5, position="identity") + scale_color_brewer(palette="Dark2")
  return(p.value)
}

set.seed(101)
#resampling method 
rf_logit <- perm_p_value(all, "L2_Logistic_Regression","Random_Forest") # different p=0.00049995
perm_p_value(all,"XGBoost", "L2_Logistic_Regression") # not different
perm_p_value(all, "XGBoost", "L2_Linear_SVM") # not different
perm_p_value(all, "RBF_SVM", "L2_Linear_SVM") # not different
perm_p_value(all, "L1_Linear_SVM", "RBF_SVM") # different p = 0.00069993
perm_p_value(all, "Decision_Tree", "RBF_SVM") # different p = 9.999e-05
perm_p_value(all, "L1_Linear_SVM", "Decision_Tree") # different p = 9.999e-05

library(ggpubr)
my_comparisons = c("Random_Forest", "L2_Logistic_Regression")

performance <- ggplot(best_performance, aes(x = fct_reorder(model, AUC), y = AUC, fill = Performance)) +
  geom_boxplot(alpha=0.5, fatten = 4) +
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
  stat_compare_means(comparisons = my_comparisons, label.y = 0.9) + 
  theme_bw() +
  theme(plot.margin=unit(c(1.1,1.1,1.1,1.1),"cm"),
        legend.justification=c(1,0),
        legend.position=c(1,0),
        #legend.position="bottom",
        legend.title = element_blank(),
        legend.background = element_rect(linetype="solid", color="black", size=0.5),
        legend.box.margin=margin(c(12,12,12, 12)),
        legend.text=element_text(size=18),
        #legend.title=element_text(size=22),
        panel.grid.major.y = element_blank(),
        panel.grid.major.x = element_line( size=0.6),
        panel.grid.minor = element_blank(),
        panel.background = element_blank(),
        text = element_text(size = 12),
        axis.text.x=element_text(size = 20, colour='black'),
        axis.text.y=element_text(size = 20, colour='black'),
        axis.title.y=element_text(size = 24),
        axis.title.x=element_text(size = 24),
        panel.border = element_rect(linetype="solid", colour = "black", fill=NA, size=1.5))

######################################################################
#-----------------------Save figure as .pdf ------------------------ #
######################################################################

ggsave("Figure_2.png", plot = performance, device = 'png', path = 'submission', width = 12, height = 9)
