setwd('H:/Machine_Learning/Otto-Group-Product-Classification-Challenge')
setwd('/Users/ivan/Work_directory/Otto-Group-Product-Classification-Challenge')
setwd('C:/Users/Ivan.Liuyanfeng/Desktop/Data_Mining_Work_Space/Otto-Group-Product-Classification-Challenge')
rm(list=ls());gc()
require(data.table);require(caret)

# train <- as.data.frame(fread(unzip('data/train.csv.zip'),stringsAsFactors = F))
# test <- as.data.frame(fread(unzip('data/test.csv.zip'),stringsAsFactors = F))
# sampleSubmission <- fread('data/sampleSubmission.csv',stringsAsFactors = F)
# save(train,test, file='data/raw_data_multi.RData')
load(file='data/raw_data.RData')

# levels(as.factor(train$target))
# dummies <- dummyVars(~target, data = train)
# target <- predict(dummies, newdata = train)
# save(target, file='data/target.RData')
# train$target <- NULL
# train <- cbind(train, target)
head(train);colnames(train)
table(train$target)

fitControl <- trainControl(method = "adaptive_cv", number = 10, repeats = 3, classProbs = TRUE,
                           adaptive = list(min = 10,alpha = 0.05,method = 'BT',complete = TRUE))
gbmGrid <-  expand.grid(shrinkage=0.1, interaction.depth=1, n.trees=150) 
fit <- train(x = train[,c(2:94)], y = as.factor(train[,95]), method ="gbm", metric ='Kappa', 
             trControl = fitControl,tuneLength = 8)#,tuneGrid = gbmGrid) #Accuracy

res <- predict(fit, newdata=test,type = "prob")
submission <- cbind(id=test$id, res)
write.csv(submission,file='../first_try_gbm.csv',row.names=F)

