######################################################################
#---------- Load .csv datato get mean traintime for 7 models-------#
######################################################################

# Read in the traintime for each split.
traintime_files <- list.files(path= '../data/process', pattern='traintime*', full.names = TRUE) 

# ------- 1. In a loop read the files as delim
# --------2. get the model name for each with get_model_name() 
#---------3. Add a column with model name to each delim
result <- list()

for(file in traintime_files){
  model_traintime <- read_files(file) %>%  
    #summarise_traintime() %>% 
    mutate(model=get_model_name(file))
  result[[length(result)+1]] <- model_traintime
}


# ------- 1. If models are L1 or L2 Linear SVM or L2 Logit then the traintime is in minutes
# --------2. If models are Random Forest or XGBoost then the traintime is in days
#---------3. If other models, the traintime is in hours.
#---------4. This loop converts all to hours.
min_fixed_result <- list() 

for(i in result){
  i$x <- i$x/3600 # The traintimes were saved as seconds we covert to hours
  min_fixed_result[[length(min_fixed_result)+1]] <- i
}
# Bind all delims with model name and traintime together
# Summarise mean traintime
traintime_df <- bind_rows(min_fixed_result) %>% 
  group_by(model) %>% 
  summarise(mean_traintime=mean(x), sd_traintime=sd(x))

# Get the sorted mean traintime for each model small to large
traintime_index <- order(traintime_df$mean_traintime) 
