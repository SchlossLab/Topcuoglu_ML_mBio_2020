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

# Load .csv data generated with modeling pipeline 
######################################################################

# Read in the walltime for each split.
walltime_files <- list.files(path= 'data/process', pattern='walltime*', full.names = TRUE) 

result <- list()

for(file in walltime_files){
  model_walltime <- read_files(file) %>%  
  #summarise_walltime() %>% 
  mutate(model=get_model_name(file))
  result[[length(result)+1]] <- model_walltime
}

min_fixed_result <- list()
for(i in result){
  i$x <- i$x/3600 # The walltimes were saved as seconds we covert to hours
  min_fixed_result[[length(min_fixed_result)+1]] <- i
}


walltime_df <- bind_rows(min_fixed_result)

######################################################################
#Plot the wall-time values for each model #
######################################################################


walltime_plot <- ggplot(walltime_df, aes(x = fct_reorder(model, x), y = x)) +
  geom_boxplot(alpha=0.7, fill="darkgoldenrod1") +
  scale_y_continuous(name = "Walltime (hours)") +
  scale_x_discrete(name = "",
                   labels=c("L2 Logistic Regression",
                     "L1 Linear SVM",
                     "L2 Linear SVM",
                     "Decision Tree",
                     "RBF SVM",
                      "Random Forest", 
                     "XGBoost")) +
  theme_bw() +
  theme(legend.title=element_text(size=22),
        panel.grid.major = element_blank(),
        panel.grid.minor = element_blank(),
        panel.background = element_blank(),
        text = element_text(size = 12),
        axis.text.x=element_text(size = 16, colour='black'),
        axis.text.y=element_text(size = 16, colour='black'),
        axis.title.y=element_text(size = 20),
        axis.title.x=element_text(size = 20), 
        panel.border = element_rect(colour = "black", fill=NA, size=1)) 

######################################################################
#-----------------------Save figure as .pdf ------------------------ #
######################################################################

ggsave("Figure_5.png", plot = walltime_plot, device = 'png', path = 'submission', width = 15, height = 10)


            