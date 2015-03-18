setwd('H:/Machine_Learning/Otto-Group-Product-Classification-Challenge')
setwd('/Users/ivan/Work_directory/Otto-Group-Product-Classification-Challenge')
setwd('C:/Users/Ivan.Liuyanfeng/Desktop/Data_Mining_Work_Space/Otto-Group-Product-Classification-Challenge')
rm(list=ls());gc()
require(caret)

load(file='data/raw_data_multi.RData')
table(train$target)

################
### Modeling ###
################
library(doMC)
registerDoMC(cores = 2)

fitControl <- trainControl(method = "none", number = 10, repeats = 3, classProbs = TRUE,
                           adaptive = list(min = 10,alpha = 0.05,method = 'BT',complete = TRUE))
gbmGrid <-  expand.grid(n.trees = 400, interaction.depth = 8, shrinkage = 0.1) 
fit <- train(x = train[,c(2:94)], y = as.factor(train[,95]), method ="gbm", metric ='Kappa', 
             trControl = fitControl,tuneLength = 8,tuneGrid = gbmGrid) #Accuracy Kappa

trellis.par.set(caretTheme())
plot(fit, metric = "Kappa")

res <- predict(fit, newdata=test,type = "prob")
source('main/2_logloss_func.R')
logloss(res)

submission <- cbind(id=test$id, res)
write.csv(submission,file='../first_try_gbm.csv',row.names=F)

######################
### Tuning Results ###
######################
# 1. gbm: n.trees = 400, interaction.depth = 8 and shrinkage = 0.1 (>8, 400)