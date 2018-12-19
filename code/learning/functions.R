# Author: Begum Topcuoglu
# Date: 2018-12-06
#
######################################################################
# Place to store useful functions that will be used repeatedly throughout
######################################################################

melt_data <-  function(data) {
  data_melt <- data %>%
    melt(measure.vars=c('cv_aucs', 'test_aucs')) %>%
    rename(AUC=value) %>%
    mutate(Performance = case_when(variable == "cv_aucs" ~ 'cross-validation', variable == "test_aucs" ~ 'testing')) %>% 
    group_by(Performance)
  return(data_melt)
}

plot_performance <- function(data) {
  data_melt <- data %>%
    mutate(model="L2 Linear SVM") %>%
    melt(measure.vars=c('cv_aucs', 'test_aucs')) %>%
    rename(AUC=value) %>%
    rename(Performance=variable) %>%
    group_by(Performance)


  performance <- ggplot(data_melt, aes(x = model,
                      y = AUC,
                      fill = Performance)) +
  geom_boxplot(alpha=0.7) +
  scale_y_continuous(name = "AUC",
                     breaks = seq(0.5, 1, 0.025),
                     limits=c(0.5, 1),
                     expand=c(0,0)) +
  scale_x_discrete(name = "") +
  theme_bw() +
  theme(legend.text=element_text(size=8),
        legend.title=element_text(size=10),
        panel.grid.major = element_blank(),
        panel.grid.minor = element_blank(),
        panel.background = element_blank(),
        text = element_text(size = 12),
        axis.text.x=element_text(size = 15,
                                 colour='black'),
        axis.text.y=element_text(size = 10,
                                 colour='black'),
        axis.title.y=element_text(size = 15),
        axis.title.x=element_text(size = 15)) +
  scale_fill_brewer(palette = "Paired") +
  geom_hline(yintercept = 0.5, linetype="dashed")+
    guides(fill=FALSE)

  return(performance)
}

plot_parameter_linear <- function(data){
  data %>% 
    group_by(Cost, Performance) %>% 
    summarise (mean_AUC = mean(AUC)) %>% 
    group_by(Performance) %>% 
    ggplot(aes(x=Cost,y=mean_AUC, color=Performance)) +
    geom_line() +
    geom_point() +scale_x_continuous(name="C (penalty)") +
    theme_bw() +
    theme(legend.text=element_text(size=18),
          legend.title=element_text(size=22),
          panel.grid.major = element_blank(),
          panel.grid.minor = element_blank(),
          panel.background = element_blank(),
          text = element_text(size = 12),
          axis.text.x=element_text(size = 12, colour='black'),
          axis.text.y=element_text(size = 12, colour='black'),
          axis.title.y=element_text(size = 13),
          axis.title.x=element_text(size = 13))
  
}


