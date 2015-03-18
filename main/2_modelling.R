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

trainIndex <- createDataPartition(train[,95], p = .7,list = FALSE)
train_df <- train[trainIndex,]
test_df  <- train[-trainIndex,]

# fitControl <- trainControl(method = "none", number = 10, repeats = 3, classProbs = TRUE,
#                            adaptive = list(min = 10,alpha = 0.05,method = 'BT',complete = TRUE))
fitControl <- trainControl(method = "none", number = 10, repeats = 5, classProbs = T, verbose = T)
gbmGrid <-  expand.grid(n.trees = 150, interaction.depth = 3, shrinkage = 0.1) 
fit <- train(x = train_df[,c(2:94)], y = as.factor(train_df[,95]), method ="gbm", metric ='Kappa', 
             trControl = fitControl,tuneGrid = gbmGrid) #Accuracy Kappa tuneLength = 8,

trellis.par.set(caretTheme())
plot(fit, metric = "Kappa")

val <- predict(fit, newdata=test_df,type = "prob")
source('main/2_logloss_func.R')
load(file='data/target.RData')
target_df <- target[-trainIndex,]
logloss(val,target_df)

res <- predict(fit, newdata=test,type = "prob")
submission <- cbind(id=test$id, res)
write.csv(submission,file='../first_try_gbm.csv',row.names=F)

######################
### Tuning Results ###
######################
# 1. gbm: n.trees = 400, interaction.depth = 8 and shrinkage = 0.1 (>8, 400)