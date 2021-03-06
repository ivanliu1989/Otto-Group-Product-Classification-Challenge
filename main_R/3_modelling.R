# setwd('H:/Machine_Learning/Otto-Group-Product-Classification-Challenge')
setwd('/Users/ivan/Work_directory/Otto-Group-Product-Classification-Challenge')
# setwd('C:/Users/Ivan.Liuyanfeng/Desktop/Data_Mining_Work_Space/Otto-Group-Product-Classification-Challenge')
rm(list=ls());gc()
require(caret)
source('main/2_logloss_func.R')
load(file='data/target.RData')
load(file='data/raw_data_multi.RData')
table(train$target)

################
### Modeling ###
################
# library(doMC)
# registerDoMC(cores = 2)

dim(train)
trainIndex <- createDataPartition(train[,95], p = .7,list = FALSE)
train_df <- train[trainIndex,]
test_df  <- train[-trainIndex,]
# train_df <-train

# fitControl <- trainControl(method = "adaptive_cv", number = 10, repeats = 5, classProbs = TRUE,
#                            adaptive = list(min = 12,alpha = 0.05,method = 'BT',complete = TRUE))
fitControl <- trainControl(method = "none", number = 10, repeats = 5, classProbs = T, verbose = T)
gbmGrid <-  expand.grid(mtry=17)  #n.trees = 50, interaction.depth = 1, shrinkage = 0.1
fit <- train(x = train_df[,c(2:94)], y = as.factor(target[trainIndex,1]), method ="rf", metric ='Kappa', 
             trControl = fitControl,do.trace=100, tuneGrid = gbmGrid)
#preProc = c("center","scale"),tuneLength = 10, repeats = 15

# trellis.par.set(caretTheme())
# plot(fit, metric = "Kappa")
val <- predict(fit, newdata=test_df[,c(2:94)],type = "prob")
target_df <- target[-trainIndex,1]
# confusionMatrix(val,target_df)
# table(apply(val,1,sum))
LogLoss(target_df,val)

res <- predict(fit, newdata=test,type = "prob")
submission <- cbind(id=test$id, res)
write.csv(submission,file='../first_try_rf.csv',row.names=F)

######################
### Tuning Results ###
######################
# 1. gbm: n.trees = 400, interaction.depth = 8 and shrinkage = 0.1 (>8, 400)
# 2. rf: mtry=17 (0.5618718 | 84 features) (0.5774515 | 94 features) | mtry=47 (0.587)

train[,83:91] <- val
test[,95:103] <- res
