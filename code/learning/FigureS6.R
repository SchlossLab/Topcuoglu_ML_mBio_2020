######################################################################
#---------- Compare training time with subsampling results -------#
######################################################################

# Load in needed functions and libraries
source('code/learning/functions.R')
# detach("package:randomForest", unload=TRUE) to run


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
# -------------------------------------------------------------------->

# Read in the traintime 100 splits.
logit_files <- list.files(path= 'data/process/subsampling', pattern='combined_traintime_L2_Logistic_Regression_.*', full.names = TRUE)
rf_files <- list.files(path= 'data/process/subsampling', pattern='combined_traintime_Random_Forest_.*', full.names = TRUE)


logit_performance <- map_df(logit_files, read_files) %>% 
  mutate(x=x/3600) %>% 
  group_by(number) 
logit_performance$number <-  as.factor(logit_performance$number)
logit_performance$number <- factor(logit_performance$number , c("15", "30", "60", "120", "245", "490"))

rf_performance <- map_df(rf_files, read_files) %>% 
  mutate(x=x/3600) %>% 
  group_by(number) 
rf_performance$number <-  as.factor(rf_performance$number)
rf_performance$number <- factor(rf_performance$number , c("15", "30", "60", "120", "245", "490"))



logit_plot <- ggplot(logit_performance, aes(x = number, y = x)) +
  geom_boxplot(alpha=0.6, fatten = 4, fill="darkred") +
  geom_smooth(method="lm", aes(color="Exp Model"), formula= (y ~ exp(x)), se=FALSE, linetype = 1) +
  scale_y_continuous(name = "L2-regularized logistic regression 
training time (hours)",
                     breaks = seq(0, 0.2, 0.05),
                     limits=c(0, 0.2),
                     expand=c(0,0)) +
  scale_x_discrete(name="") +
  theme_bw() +
  theme(plot.margin=unit(c(1.8,3.4,1.8,3.4),"mm"),
        panel.grid.major.y = element_blank(),
        panel.grid.major.x = element_blank(),
        panel.grid.minor = element_blank(),
        panel.background = element_blank(),
        text = element_text(size = 14),
        axis.text.y=element_text(size = 14, colour='black'),
        axis.title.y=element_text(size = 16),
        axis.text.x=element_blank(),
        panel.border = element_rect(linetype="solid", colour = "black", fill=NA, size=1.5))

rf_plot <- ggplot(rf_performance, aes(x = number, y = x)) +
  geom_boxplot(alpha=0.6, fatten = 4, fill="darkred") +
  scale_y_continuous(name = "Random forest 
training time (hours)",
                     breaks = seq(0, 100, 25),
                     limits=c(0, 100),
                     expand=c(0,0)) +
  scale_x_discrete(name="") +
  theme_bw() +
  theme(plot.margin=unit(c(1.7,3.2,1.7,3.2),"mm"),
        panel.grid.major.y = element_blank(),
        panel.grid.major.x = element_blank(),
        panel.grid.minor = element_blank(),
        panel.background = element_blank(),
        text = element_text(size = 14),
        axis.text.y=element_text(size = 14, colour='black'),
        axis.title.y=element_text(size = 16),
        axis.text.x=element_text(size = 14, colour='black'),
        panel.border = element_rect(linetype="solid", colour = "black", fill=NA, size=1.5))

#combine with cowplot

plots <- plot_grid(logit_plot, rf_plot, labels = c("A", "B"), align = 'v', ncol = 1)

ggdraw(add_sub(plots, "Sample size (N)", vpadding=grid::unit(0,"lines"), y=5, x=0.6, vjust=4.75, size=15))

ggsave("Figure_S6.png", plot = last_plot(), device = 'png', path = 'submission', width = 6, height = 9.5)
