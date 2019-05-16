######################################################################
#---------- Load .csv datato get mean walltime for 7 models-------#
######################################################################

# Read in the walltime for each split.
walltime_files <- list.files(path= '../data/process', pattern='walltime*', full.names = TRUE) 

# ------- 1. In a loop read the files as delim
# --------2. get the model name for each with get_model_name() 
#---------3. Add a column with model name to each delim
result <- list()

for(file in walltime_files){
  model_walltime <- read_files(file) %>%  
    #summarise_walltime() %>% 
    mutate(model=get_model_name(file))
  result[[length(result)+1]] <- model_walltime
}


# ------- 1. If models are L1 or L2 Linear SVM or L2 Logit then the walltime is in minutes
# --------2. If models are Random Forest or XGBoost then the walltime is in days
#---------3. If other models, the walltime is in hours.
#---------4. This loop converts all to hours.
min_fixed_result <- list() 

for(i in result){
  i$x <- i$x/3600 # The walltimes were saved as seconds we covert to hours
  min_fixed_result[[length(min_fixed_result)+1]] <- i
}
# Bind all delims with model name and walltime together
# Summarise mean walltime
walltime_df <- bind_rows(min_fixed_result) %>% 
  group_by(model) %>% 
  summarise(mean_walltime=mean(x), sd_walltime=sd(x))

# Get the sorted mean walltime for each model small to large
walltime_index <- order(walltime_df$mean_walltime) 
