# Author: Begum Topcuoglu
# Date: 2018-08-18
#
######################################################################
# This script plots Figure S3:
#   1. cvAUC (medians of 100 repeats for the best hp) of 100 datasplits
#   2. How subsampling changes the performance
######################################################################

######################################################################
# Load in needed functions and libraries
source('code/learning/functions.R')
# detach("package:randomForest", unload=TRUE) to run
######################################################################
# -------------------- Read files ------------------------------------>
grab_number <- function(file){
  regmatches(file, regexpr("[0-9].*[0-9]", file)) %>% 
    str_extract("[^_]*$")
}
# Read in files as delim that are saved in a list with a pattern
read_files <- function(filenames){
  for(file in filenames){
    number <- grab_number(file)
    # Read the files generated in main.R
    # These files have cvAUCs and testAUCs for 100 data-splits
    data <- read.delim(file, header=T, sep=',') %>% 
      mutate(number = number)
  }
  return(data)
}
######################################################################
# Load .csv data generated with modeling pipeline
######################################################################

# Read in the cvAUCs, testAUCs for 100 splits.
logit_files <- list.files(path= 'data/process/subsampling', pattern='combined_best_hp_results_L2_Logistic_Regression_.*', full.names = TRUE)

logit_performance <- map_df(logit_files, read_files) %>% 
  melt_data() %>% 
  unite_("model_number", c("model","number")) %>% 
  filter(Performance != "Testing") %>% 
  group_by(model_number) 
  
logit_performance$model_number <-  as.factor(logit_performance$model_number)


logit_performance$model_number <- factor(logit_performance$model_number , c("L2_Logistic_Regression_490", "L2_Logistic_Regression_245", "L2_Logistic_Regression_120", "L2_Logistic_Regression_60", "L2_Logistic_Regression_30", "L2_Logistic_Regression_15"))

rf_files <- list.files(path= 'data/process/subsampling', pattern='combined_best_hp_results_Random_Forest_.*', full.names = TRUE)

  #filter(Performance != "testing") 

rf_performance <- map_df(rf_files, read_files) %>% 
  melt_data() %>% 
  unite_("model_number", c("model","number")) %>% 
  filter(Performance != "Testing")

rf_performance$model_number <-  as.factor(rf_performance$model_number)

rf_performance$model_number <- factor(rf_performance$model_number , c("Random_Forest_490", "Random_Forest_245", "Random_Forest_120", "Random_Forest_60", "Random_Forest_30", "Random_Forest_15"))
######################################################################
#Plot the AUC values for cross validation  for each model #
######################################################################


logit_plot <- ggplot(logit_performance, aes(x = model_number, y = AUC)) +
  geom_boxplot(alpha=0.5, fatten = 4, fill="blue4") +
  geom_hline(yintercept = 0.5, linetype="dashed") +
  coord_flip() +
  scale_y_continuous(name = "",
                     breaks = seq(0, 1, 0.1),
                     limits=c(0, 1),
                     expand=c(0,0)) +
  scale_x_discrete(name = expression(paste(L[2], "-regularized logistic regression (N)")), 
                   labels=c("490",
                            "245", 
                            "120",
                            "60",
                            "30",
                            "15")) +
  theme_bw() +
  theme(plot.margin=unit(c(1.5,3,1.5,3),"mm"),
        panel.grid.major.y = element_blank(),
        panel.grid.major.x = element_line( size=0.6),
        panel.grid.minor = element_blank(),
        panel.background = element_blank(),
        text = element_text(size = 14),
        axis.text.y=element_text(size = 14, colour='black'),
        axis.title.y=element_text(size = 16),
        axis.text.x=element_blank(),
        panel.border = element_rect(linetype="solid", colour = "black", fill=NA, size=1.5))

rf_plot <- ggplot(rf_performance, aes(x = model_number, y = AUC)) +
  geom_boxplot(alpha=0.5, fatten = 4, fill="blue4") +
  geom_hline(yintercept = 0.5, linetype="dashed") +
  coord_flip() +
  scale_y_continuous(name = "",
                     breaks = seq(0, 1, 0.1),
                     limits=c(0, 1),
                     expand=c(0,0)) +
  scale_x_discrete(name = "Random forest (N)", 
                   labels=c("490",
                            "245", 
                            "120",
                            "60",
                            "30",
                            "15")) +
  theme_bw() +
  theme(plot.margin=unit(c(1.5,3,1.5,3),"mm"),
        panel.grid.major.y = element_blank(),
        panel.grid.major.x = element_line( size=0.6),
        panel.grid.minor = element_blank(),
        panel.background = element_blank(),
        text = element_text(size = 14),
        axis.text.y=element_text(size = 14, colour='black'),
        axis.title.y=element_text(size = 16),
        axis.text.x=element_text(size = 14, colour='black'),
        panel.border = element_rect(linetype="solid", colour = "black", fill=NA, size=1.5))

######################################################################
#-----------------------Save figure as .pdf ------------------------ #
######################################################################

#combine with cowplot

plots <- plot_grid(logit_plot, rf_plot, labels = c("A", "B"), align = 'v', ncol = 1)

ggdraw(add_sub(plots, "Cross-validation AUROC", vpadding=grid::unit(0,"lines"), y=5, x=0.6, vjust=4.75, size=15))

ggsave("Figure_S4.png", plot = last_plot(), device = 'png', path = 'submission', width = 6, height = 9.2)

