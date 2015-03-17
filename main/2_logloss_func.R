load(file='data/target.RData')
logloss <- function(pred, target = target){
    n <- nrow(pred)
    m <- 9
    p <- 0
    for(i in 1:n){
        for(j in 1:m){
            p <- target[i,j] * max(min(pred[i,j],1−10^−15),10^−15) + p
        }
    }
    return(-1/n*p)
}