
######################################################################
#----------------- Read in necessary libraries -------------------#
######################################################################
deps = c("reshape2", "cowplot", "ggplot2","knitr","rmarkdown","vegan","gtools", "tidyverse");
for (dep in deps){
  if (dep %in% installed.packages()[,"Package"] == FALSE){
    install.packages(as.character(dep), quiet=TRUE);
  }
  library(dep, verbose=FALSE, character.only=TRUE)
}


# Read in hyper-parameter AUCs generated from L2 logistic regression model for all samples:
logit <- read.delim('data/process/L2_Logistic_Regression_aucs_hps_R.tsv', header=T, sep='\t')

logit %>%
  summarise(cv_mean=mean(cv_aucs), 
            test_mean=mean(test_aucs), 
            cv_sd=sd(cv_aucs), 
            test_sd=sd(test_aucs))

logit_melt <- logit %>%
  mutate(model="L2 Logistic Regression") %>% 
  melt(measure.vars=c('cv_aucs', 'test_aucs')) %>% 
  rename(AUC=value) %>% 
  rename(Performance=variable) %>% 
  group_by(Performance)
  

performance <- ggplot(logit_melt, aes(x = model, 
                       y = AUC, 
                       fill = Performance)) +
  geom_boxplot(alpha=0.7) +
  scale_y_continuous(name = "AUC",
                     breaks = seq(0.5, 1, 0.02),
                     limits=c(0.5, 1), expand=c(0,0)) +
  scale_x_discrete(name = "") +
  theme_bw() +
  theme(legend.text=element_text(size=8),
        legend.title=element_text(size=10),
        panel.grid.major = element_blank(),
        panel.grid.minor = element_blank(),
        panel.background = element_blank(),
        text = element_text(size = 12),
        axis.text.x=element_text(size = 15, colour='black'),
        axis.text.y=element_text(size = 10, colour='black'),
        axis.title.y=element_text(size = 15),
        axis.title.x=element_text(size = 15)) +
  scale_fill_brewer(palette = "Paired") +
  geom_hline(yintercept = 0.5, linetype="dashed")



parameter <- ggplot(logit, aes(x = Cost, y = cv_aucs))+
  geom_point()+
  stat_summary(aes(y = cv_aucs,group=1), fun.y=mean, colour="black", geom="line",group=1) + 
  theme_bw() +
  theme(legend.text=element_text(size=18),
        legend.title=element_text(size=22),
        panel.grid.major = element_blank(),
        panel.grid.minor = element_blank(),
        panel.background = element_blank(),
        text = element_text(size = 12),
        axis.text.x=element_text(size = 10, colour='black'),
        axis.text.y=element_text(size = 10, colour='black'),
        axis.title.y=element_text(size = 15),
        axis.title.x=element_text(size = 15)) +
  scale_y_continuous(name = "Cross-validation AUC",
                     breaks = seq(0.5, 1, 0.02),
                     limits=c(0.5, 1), expand=c(0,0))


plot_grid(performance, parameter, labels = c("A", "B"))

######################################################################
#-----------------------Save figure as .pdf ------------------------ #
######################################################################

ggsave("L2_Logistic_Regression_comparison.pdf", plot = last_plot(), device = 'pdf', path = 'results/figures', width = 15, height = 10)
  
  