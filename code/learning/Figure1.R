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
# Load .tsv data generated with modeling pipeline 
######################################################################

# Read in the cvAUCs, testAUCs for 100 splits.
best_files <- list.files(path= 'data/process', pattern='combined_best.*', full.names = TRUE)


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

ggsave("Figure_1.pdf", plot = performance, device = 'pdf', path = 'results/figures', width = 15, height = 10)

