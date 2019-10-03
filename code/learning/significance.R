library(tidyverse)
source('code/learning/functions.R')
######################################################################
#--------- Load cv and test AUROCs of 7 models for 100 datasplits--------#
######################################################################

# Read in the cvAUCs, testAUCs for 100 splits.
best_files <- list.files(path= 'data/process', pattern='combined_best.*', full.names = TRUE)
all <- map_df(best_files, read_files) 

histogram_p_value(all, "L2_Logistic_Regression","Random_Forest")

# Resampling test of means from two groups for L2-logistic regression and random forest

model_name_1 <- "Decision_Tree"
model_name_2 <- "Random_Forest"
histogram_p_value <- function(data, model_name_1, model_name_2){
  data_1 <- all %>% filter(model==model_name_1)
  data_2 <- all %>% filter(model==model_name_2)
  library(mosaic)
  observed_diff <- base::mean(data_2$test_aucs) - mean(data_1$test_aucs)
  resampled_data1 <- do(1000) * resample(data_1) 
  resampled_data2 <- do(1000) * resample(data_2)
  detach("package:mosaic", unload=TRUE)
  d1 <- resampled_data1 %>% 
    spread(.index, test_aucs) %>% 
    select(-cv_aucs, -model, -orig.id, -.row) %>% 
    colMeans(na.rm=TRUE)
  d2 <- resampled_data2 %>% 
    spread(.index, test_aucs) %>% 
    select(-cv_aucs, -model, -orig.id, -.row) %>% 
    colMeans(na.rm=TRUE)
  resampled_diff <- d2-d1
  ggplot() + aes(resampled_diff, color=resampled_diff>=observed_diff) + geom_histogram(fill="white", alpha=0.5, position="identity") + scale_color_brewer(palette="Dark2")
  
  
  plot <- ggplot() + 
    aes(resampled_diff, color=resamples_diff>=observed_diff) +
    geom_histogram(aes(y=..density..), colour="black", fill="white", bins=30)+
    geom_density(alpha=.2, fill="blue") +
    geom_vline(xintercept = 0, color="red", size=1) +
    geom_hline(yintercept = 0, color="red", size=1) +
    #geom_vline(aes(xintercept=mean(differences)),
    #           color="blue", linetype="dashed") + 
    coord_cartesian(xlim=c(-0.2,0.2)) +
    scale_x_continuous(name= "The difference between random forest and L2-regularized
                       logistic regression AUROC values of each datasplit") +
    scale_y_continuous(expand=c(0,0)) +
    theme(axis.line = element_blank(),
          panel.background = element_blank(),
          axis.text.y = element_blank(),
          axis.title.y = element_blank(),
          axis.ticks.y = element_blank(), 
          axis.text.x=element_text(size = 12, colour='black'),
          axis.title.x=element_text(size = 14, colour='black')) +
    annotate("text", x= 0.15, y= 6, label = paste0("p-value = ", p_value))
  
  



perm_p_value(all, "Random_Forest", "L2_Logistic_Regression")
