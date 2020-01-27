# Author: Begum Topcuoglu
# Date: 2018-02-12
#
######################################################################
# This script plots Figure 5:
#   The training times of 7 models
######################################################################

######################################################################
# Load in needed functions and libraries
source('code/learning/functions.R')
# detach("package:randomForest", unload=TRUE) to run
######################################################################

# Load .csv data generated with modeling pipeline 
######################################################################

# Read in the traintime for each split.
traintime_files <- list.files(path= 'data/process', pattern='traintime*', full.names = TRUE) 

result <- list()

for(file in traintime_files){
  model_traintime <- read_files(file) %>%  
  #summarise_traintime() %>% 
  mutate(model=get_model_name(file))
  result[[length(result)+1]] <- model_traintime
}

min_fixed_result <- list()
for(i in result){
  i$x <- i$x/3600 # The traintimes were saved as seconds we covert to hours
  min_fixed_result[[length(min_fixed_result)+1]] <- i
}


traintime_df <- bind_rows(min_fixed_result)

######################################################################
#Plot the wall-time values for each model #
######################################################################


traintime_plot <- ggplot(traintime_df, aes(x = fct_reorder(model, x), y = x)) +
  geom_boxplot(fill="#0072B2", alpha=0.5, fatten = 1.5) +
  coord_flip() +
  scale_y_continuous(name = "Training time (hours)") +
  scale_x_discrete(name = "",
                   labels=c(expression(paste(L[2], "-regularized logistic regression")),
                            expression(paste(L[1], "-regularized linear SVM")),
                            expression(paste(L[2], "-regularized linear SVM")),
                            "Decision tree",
                            "SVM with radial basis kernel",
                            "Random forest", 
                            "XGBoost")) +
  theme_bw() +
  theme(legend.title=element_text(size=15),
        panel.grid.major = element_blank(),
        panel.grid.minor = element_blank(),
        panel.background = element_blank(),
        text = element_text(size = 10),
        axis.text.x=element_text(size = 8, colour='black'),
        axis.text.y=element_text(size = 8, colour='black'),
        axis.title.y=element_text(size = 10),
        axis.title.x=element_text(size = 10), 
        panel.border = element_rect(colour = "black", fill=NA, size=1)) 

######################################################################
#-----------------------Save figure as .pdf ------------------------ #
######################################################################

ggsave("Figure_5.tiff", plot = traintime_plot, device = 'tiff', path = 'submission', width = 4, height = 2, dpi=300)


            