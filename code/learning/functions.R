# Author: Begum Topcuoglu
# Date: 2018-12-06
#
######################################################################
# Place to store useful functions that will be used repeatedly throughout
######################################################################

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
                     breaks = seq(0.5, 1, 0.02),
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
  geom_hline(yintercept = 0.5, linetype="dashed")

  return(performance)
}

plot_parameter <- function(data){
  parameter <- ggplot(data,
                      aes(x = Cost, y = cv_aucs)) +
    geom_point() +
    stat_summary(aes(y = cv_aucs,group=1),
                 fun.y=mean,
                 colour="black",
                 geom="line",
                 group=1) +
    theme_bw() +
    theme(legend.text=element_text(size=18),
          legend.title=element_text(size=22),
          panel.grid.major = element_blank(),
          panel.grid.minor = element_blank(),
          panel.background = element_blank(),
          text = element_text(size = 12),
          axis.text.x=element_text(size = 10,
                                   colour='black'),
          axis.text.y=element_text(size = 10,
                                   colour='black'),
          axis.title.y=element_text(size = 15),
          axis.title.x=element_text(size = 15)) +
    scale_y_continuous(name = "Cross-validation AUC",
                       breaks = seq(0.5, 1, 0.02),
                       limits=c(0.5, 1), expand=c(0,0))
  return(parameter)
}
