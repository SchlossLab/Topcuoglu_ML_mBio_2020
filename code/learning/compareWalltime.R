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

# Read in the cvAUCs, testAUCs for 100 splits.
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
  if((sum(i$model=="L1_Linear_SVM")==100 || sum(i$model=="L2_Linear_SVM")==100 || sum(i$model=="L2_Logistic_Regression")==100)){
    i$x <- i$x/60 # The walltimes were saved as minutes
    print(i) # We convert these to hours by diving with 60
  }
  else{
    print("not minutes")
  }
  min_fixed_result[[length(min_fixed_result)+1]] <- i
}


walltime_df <- bind_rows(min_fixed_result)

######################################################################
#Plot the wall-time values for each model #
######################################################################


walltime_plot <- ggplot(walltime_df, aes(x = fct_reorder(model, x, fun = median, .asc =TRUE), y = x)) +
  geom_boxplot(alpha=0.7) +
  scale_y_continuous(name = "Walltime") +
  scale_x_discrete(name = "") +
  theme_bw() +
  theme(legend.title=element_text(size=22),
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

ggsave("Figure_3.pdf", plot = walltime_plot, device = 'pdf', path = 'results/figures', width = 15, height = 10)


            