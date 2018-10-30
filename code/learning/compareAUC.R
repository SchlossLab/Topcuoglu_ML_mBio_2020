deps = c("beeswarm", "ggplot2","pROC", "caret","knitr","rmarkdown","vegan","gtools", "tidyverse");
for (dep in deps){
  if (dep %in% installed.packages()[,"Package"] == FALSE){
    install.packages(as.character(dep), quiet=TRUE);
  }
  library(dep, verbose=FALSE, character.only=TRUE)
}

# Read in metadata and select only sample Id and diagnosis columns
logit <- read.delim('data/process/L2_Logistic_Regression.tsv', header=T, sep='\t') %>%
  select(level_1, AUC) %>% 
  rename(Performance = level_1)

logit$Performance <- factor(logit$Performance,
                           labels = c("Cross-validation", "Testing"))


ggplot(logit, aes(x = Performance, y = AUC, fill = Performance)) +
  geom_boxplot(alpha=0.7) +
  scale_y_continuous(name = "AUC",
                     breaks = seq(0, 1, 0.2),
                     limits=c(0, 1)) +
  scale_x_discrete(name = "") +
  theme_bw() +
  theme(panel.grid.major = element_blank(), 
        panel.grid.minor = element_blank(),
        panel.background = element_blank(),
    plot.title = element_text(size = 14, family = "Tahoma", face = "bold"),
        text = element_text(size = 12, family = "Tahoma"),
        axis.title = element_text(face="bold"),
        axis.text.x=element_text(size = 11)) +
  scale_fill_brewer(palette = "Accent")


  
