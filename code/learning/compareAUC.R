deps = c("ggplot2","knitr","rmarkdown","vegan","gtools", "tidyverse");
for (dep in deps){
  if (dep %in% installed.packages()[,"Package"] == FALSE){
    install.packages(as.character(dep), quiet=TRUE);
  }
  library(dep, verbose=FALSE, character.only=TRUE)
}


logit <- read.delim('data/process/L2_Logistic_Regression.tsv', header=T, sep='\t') %>%
  select(level_1, AUC) %>% 
  rename(Performance = level_1) %>% 
  mutate(model="L2-Logistic Regression")

logit$Performance <- factor(logit$Performance,
                           labels = c("Cross-validation", "Testing"))

l1svm <- read.delim('data/process/L1_SVM_Linear_Kernel.tsv', header=T, sep='\t') %>%
  select(level_1, AUC) %>% 
  rename(Performance = level_1) %>% 
  mutate(model="L1-SVM Linear Kernel") 


l2svm <- read.delim('data/process/L2_SVM_Linear_Kernel.tsv', header=T, sep='\t') %>%
  select(level_1, AUC) %>% 
  rename(Performance = level_1) %>% 
  mutate(model="L2-SVM Linear Kernel")%>% 
  bind_rows(logit, l1svm) %>% 
  group_by(model)



ggplot(l2svm, aes(x = model, y = AUC, fill = Performance)) +
  geom_boxplot(alpha=0.7) +
  scale_y_continuous(name = "AUC",
                     breaks = seq(0.4, 1, 0.05),
                     limits=c(0.4, 1)) +
  scale_x_discrete(name = "") +
  theme_bw() +
  theme(panel.grid.major = element_blank(), 
        panel.grid.minor = element_blank(),
        panel.background = element_blank(),
        text = element_text(size = 12),
        axis.text.x=element_text(size = 15, colour='black'),
        axis.text.y=element_text(size = 12, colour='black'), 
        axis.title.y=element_text(size = 20), 
        axis.title.x=element_text(size = 20)) +
  scale_fill_brewer(palette = "Paired")

ggsave("AUC_comparison.pdf", plot = last_plot(), device = 'pdf', path = 'results/figures', width = 10, height = 10)

  
