# Validation of spatial predictions and uncertainty quantifications
# Using leave-location-out cross-validation (LLOCV)
#
# Author: Gabor Szatmari
# E-mail: szatmari.gabor@atk.hun-ren.hu
#
# Please cite:
# Szatmári et al. (2024): Gridded, temporally referenced spatial information on soil organic carbon for Hungary. Scientific Data (submitted manuscript)


# 1. Packages ----
library(caret)
library(CAST)
library(ranger)
library(parallel)
library(ie2misc)


# 2. Cross-validation (5 times repeated 10-fold LLOCV) ----
points <- readRDS("00_Files/Points_ready_to_use.rds") # read the regression matrix with SOC data


## 2.1. Create folds ====
set.seed(1988)
indicies <- CreateSpacetimeFolds(x= points,
                                 spacevar = "POINT",
                                 k = 10)[[1]]

folds <- list()
folds[[1]] <- indicies

set.seed(1234)
for(i in 2:5){
  folds[[i]] <- CreateSpacetimeFolds(x= points,
                                     spacevar = "POINT",
                                     k = 10)[[1]]
}

folds <- unlist(folds, recursive=FALSE)


## 2.2. Fit QRF model for each fold ====
for(i in 1:length(folds)){
  train <- points[folds[[i]],]
  
  # Fine-tune and fit QRF models
  indicies <- CreateSpacetimeFolds(x= train,
                                   spacevar = "POINT",
                                   k = 10)
  
  t.control <- trainControl(method="cv",
                            index = indicies$index)
  
  t.grid <- expand.grid(mtry=c(1:4, seq(5, 65, by=5)),
                        splitrule="variance",
                        min.node.size=c(5))
  
  ranger <- list()
  
  ranger[[1]] <- train(x=train[,c(8:74)],
                       y=train$SOC.c,
                       method="ranger",
                       importance="impurity",
                       num.trees=500,
                       trControl=t.control,
                       tuneGrid=t.grid,
                       verbose=TRUE,
                       quantreg=TRUE,
                       num.threads=(detectCores()-1)) # for SOC content
  
  ranger[[2]] <- train(x=train[,c(8:74)],
                       y=train$SOC.s,
                       method="ranger",
                       importance="impurity",
                       num.trees=500,
                       trControl=t.control,
                       tuneGrid=t.grid,
                       verbose=TRUE,
                       quantreg=TRUE,
                       num.threads=(detectCores()-1)) # for SOC stock
  
  ranger[[3]] <- train(x=train[,c(8:74)],
                       y=train$SOC.d,
                       method="ranger",
                       importance="impurity",
                       num.trees=500,
                       trControl=t.control,
                       tuneGrid=t.grid,
                       verbose=TRUE,
                       quantreg=TRUE,
                       num.threads=(detectCores()-1)) # for SOC density
  
  if(i < 10){
    saveRDS(ranger, file = paste0("00_Files/Validation/ranger_models/ranger_models_0",i,".rds"))
  } else {
    saveRDS(ranger, file = paste0("00_Files/Validation/ranger_models/ranger_models_",i,".rds"))
  }
  
  rm(train, indicies, test, t.control, t.grid, ranger)
  gc()
  
}


## 2.3. Perform cross-validation ====
year=1992 # set the year of interest

for(i in 1:length(folds)){
  test <- points[-folds[[i]],]
  
  # read QRF models fitted previously
  if(i < 10){
    file <- list.files(path=paste0("00_Files/Validation/ranger_models/"),
                       pattern=paste0("ranger_models_0",i, ".*\\.rds$"),
                       full.names=TRUE)
  } else {
    file <- list.files(path=paste0("00_Files/Validation/ranger_models/"),
                       pattern=paste0("ranger_models_",i, ".*\\.rds$"),
                       full.names=TRUE)
  }
  
  ranger <- readRDS(file)
  
  # Predict using QRF models
  test.yr <- test[which(test$YEAR==year),]
  test.yr$pred.SOC.c <- predict(ranger[[1]], test.yr, num.threads=(detectCores()-1)) # predict for SOC content
  test.yr$pred.SOC.s <- predict(ranger[[2]], test.yr, num.threads=(detectCores()-1)) # predict for SOC stock
  test.yr$pred.SOC.d <- predict(ranger[[3]], test.yr, num.threads=(detectCores()-1)) # predict for SOC density
  
  # Quantify uncertainty using ranger models
  uncer.SOC.c <- data.frame(predict(ranger[[1]]$finalModel, test[which(test$YEAR==year),], num.threads=(detectCores()-1), type="quantiles", quantiles=seq(1,0, by=-0.01)[2:100])$predictions) # quantiles for SOC content
  names(uncer.SOC.c) <- paste0("q.", seq(1,0, by=-0.01)[2:100])
  
  uncer.SOC.s <- data.frame(predict(ranger[[2]]$finalModel, test[which(test$YEAR==year),], num.threads=(detectCores()-1), type="quantiles", quantiles=seq(1,0, by=-0.01)[2:100])$predictions) # quantiles for SOC stock
  names(uncer.SOC.s) <- paste0("q.", seq(1,0, by=-0.01)[2:100])
  
  uncer.SOC.d <- data.frame(predict(ranger[[3]]$finalModel, test[which(test$YEAR==year),], num.threads=(detectCores()-1), type="quantiles", quantiles=seq(1,0, by=-0.01)[2:100])$predictions) # quantiles for SOC density
  names(uncer.SOC.d) <- paste0("q.", seq(1,0, by=-0.01)[2:100])
  
  error <- data.frame(POINT=test.yr$POINT,
                      X=test.yr$X,
                      Y=test.yr$Y,
                      year=year,
                      obs.SOC.c=test.yr$SOC.c,
                      obs.SOC.s=test.yr$SOC.s,
                      obs.SOC.d=test.yr$SOC.d,
                      pred.SOC.c=test.yr$pred.SOC.c,
                      pred.SOC.s=test.yr$pred.SOC.s,
                      pred.SOC.d=test.yr$pred.SOC.d,
                      fold=i)
  
  uncertainty <- list()
  uncertainty[[1]] <- uncer.SOC.c
  uncertainty[[2]] <- uncer.SOC.s
  uncertainty[[3]] <- uncer.SOC.d
  names(uncertainty) <- c("SOC.c", "SOC.s", "SOC.d")
  
  if(i < 10){
    saveRDS(error, file = paste0("00_Files/Validation/",year,"/error_0", i,".rds"))
    saveRDS(uncertainty, file = paste0("00_Files/Validation/",year,"/uncertainty_0", i,".rds"))
  } else {
    saveRDS(error, file = paste0("00_Files/Validation/",year,"/error_", i,".rds"))
    saveRDS(uncertainty, file = paste0("00_Files/Validation/",year,"/uncertainty_", i,".rds"))
  }
  
  rm(file, test, test.yr, ranger, error, uncertainty, uncer.SOC.c, uncer.SOC.s, uncer.SOC.d)
  
}


# 3. Assessment ----

## 3.1. Error ====
year=1992 # set the year of interest

files <- list.files(path=paste0("00_Files/Validation/", year,"/"),
                    pattern=paste0("error.*.\\.rds$"),
                    full.names=TRUE) # read error tables

temp <- list()

for(i in 1:50){
  temp[[i]] <- readRDS(files[i])
}

temp <- do.call(rbind, temp)

temp$fold <- as.factor(temp$fold)

temp$error.SOC.c <- temp$pred.SOC.c - temp$obs.SOC.c # compute error for SOC content
temp$error.SOC.s <- temp$pred.SOC.s - temp$obs.SOC.s # compute error for SOC stock
temp$error.SOC.d <- temp$pred.SOC.d - temp$obs.SOC.d # compute error for SOC density

temp$sq.error.SOC.c <- temp$error.SOC.c^2 # compute sq.error for SOC content
temp$sq.error.SOC.s <- temp$error.SOC.s^2 # compute sq.error for SOC content
temp$sq.error.SOC.d <- temp$error.SOC.d^2 # compute sq.error for SOC content

error <- data.frame(fold=c(1:50),
                    ME.SOC.c=tapply(temp$error.SOC.c, INDEX = temp$fold, FUN=mean),
                    ME.SOC.s=tapply(temp$error.SOC.s, INDEX = temp$fold, FUN=mean),
                    ME.SOC.d=tapply(temp$error.SOC.d, INDEX = temp$fold, FUN=mean),
                    RMSE.SOC.c=sqrt(tapply(temp$sq.error.SOC.c, INDEX = temp$fold, FUN=mean)),
                    RMSE.SOC.s=sqrt(tapply(temp$sq.error.SOC.s, INDEX = temp$fold, FUN=mean)),
                    RMSE.SOC.d=sqrt(tapply(temp$sq.error.SOC.d, INDEX = temp$fold, FUN=mean))) # compute ME and RMSE for each SOC property

for(i in 1:50){
  temp2 <- temp[which(temp$fold == i),]
  
  if(i==1){
    MEC.SOC.c <- ie2misc::vnse(predicted = temp2$pred.SOC.c, observed = temp2$obs.SOC.c)
    MEC.SOC.s <- ie2misc::vnse(predicted = temp2$pred.SOC.s, observed = temp2$obs.SOC.s)
    MEC.SOC.d <- ie2misc::vnse(predicted = temp2$pred.SOC.d, observed = temp2$obs.SOC.d)
  } else {
    MEC.SOC.c <- c(MEC.SOC.c, ie2misc::vnse(predicted = temp2$pred.SOC.c, observed = temp2$obs.SOC.c))
    MEC.SOC.s <- c(MEC.SOC.s, ie2misc::vnse(predicted = temp2$pred.SOC.s, observed = temp2$obs.SOC.s))
    MEC.SOC.d <- c(MEC.SOC.d, ie2misc::vnse(predicted = temp2$pred.SOC.d, observed = temp2$obs.SOC.d))
  }
  
  rm(temp2)

} # compute MEC for each SOC property

temp2 <- cbind(MEC.SOC.c, MEC.SOC.s)
temp2 <- cbind(temp2, MEC.SOC.d); rm(MEC.SOC.c, MEC.SOC.s, MEC.SOC.d)

error <- data.frame(error, temp2); rm(temp2)
error$year = year

boxplot(error[,c(2:10)]) # quickly check the results

saveRDS(error, file=paste0("00_Files/Validation/Error_",year,".rds")) # save the computed ME, RMSE and MEC values


## 3.2. Uncertainty ====
year=1992 # set the year of interest

files <- list.files(path=paste0("00_Files/Validation/", year,"/"),
                    pattern=paste0("uncertainty.*.\\.rds$"),
                    full.names=TRUE) # read quantiles

temp1 <- list()

for(i in 1:length(files)){
  temp1[[i]] <- readRDS(files[i])
}

files <- list.files(path=paste0("00_Files/Validation/", year,"/"),
                    pattern=paste0("error.*.\\.rds$"),
                    full.names=TRUE) # read error tables

temp2 <- list()

for(i in 1:length(files)){
  temp2[[i]] <- readRDS(files[i])
}

q <- seq(1,0, by=-0.01)[2:100] # quantiles used

for(i in 1:49){ # symmetric prediction intervals
  if(i==1){
    pi <- q[i] - q[(length(q) + 1) - i]
  } else {
    pi <- c(pi, q[i] - q[(length(q) + 1) - i])
  }
}


### 3.2.1. Compile PICP plots ====
temp3 <-list() # container
temp3[[1]] <- matrix(NA, ncol=51, nrow=length(pi)); temp3[[1]][,1] = pi # matrix for SOC content
temp3[[2]] <- matrix(NA, ncol=51, nrow=length(pi)); temp3[[2]][,1] = pi # matrix for SOC stock
temp3[[3]] <- matrix(NA, ncol=51, nrow=length(pi)); temp3[[3]][,1] = pi # matrix for SOC density

for(i in 1:50){ # SOC content
  
  for(j in 1:length(pi)){
    
    temp3[[1]][j,(i+1)] <- sum(ifelse(temp1[[i]]$SOC.c[,j] >= temp2[[i]]$obs.SOC.c & temp2[[i]]$obs.SOC.c > temp1[[i]]$SOC.c[,(ncol(temp1[[i]]$SOC.c) + 1) - j], yes=1, no=0))/nrow(temp1[[i]]$SOC.c)
    
  }
  
}

for(i in 1:50){ # SOC stock
  
  for(j in 1:length(pi)){
    
    temp3[[2]][j,(i+1)] <- sum(ifelse(temp1[[i]]$SOC.s[,j] >= temp2[[i]]$obs.SOC.s & temp2[[i]]$obs.SOC.s > temp1[[i]]$SOC.s[,(ncol(temp1[[i]]$SOC.s) + 1) - j], yes=1, no=0))/nrow(temp1[[i]]$SOC.s)
    
  }
  
}

for(i in 1:50){ # SOC density
  
  for(j in 1:length(pi)){
    
    temp3[[3]][j,(i+1)] <- sum(ifelse(temp1[[i]]$SOC.d[,j] >= temp2[[i]]$obs.SOC.d & temp2[[i]]$obs.SOC.d > temp1[[i]]$SOC.d[,(ncol(temp1[[i]]$SOC.d) + 1) - j], yes=1, no=0))/nrow(temp1[[i]]$SOC.d)
    
  }
  
}

temp3[[1]] <- data.frame(temp3[[1]]); names(temp3[[1]]) <- c("theo", paste0("fold.", 1:50))
temp3[[2]] <- data.frame(temp3[[2]]); names(temp3[[2]]) <- c("theo", paste0("fold.", 1:50))
temp3[[3]] <- data.frame(temp3[[3]]); names(temp3[[3]]) <- c("theo", paste0("fold.", 1:50))

names(temp3) <- c("SOC.c", "SOC.s", "SOC.d")

par(mfrow=c(1,3)) # quickly check the results
plot(pi, apply(temp3$SOC.c[,2:51], 1, mean), main="SOC content")
abline(0,1, col="red")
plot(pi, apply(temp3$SOC.s[,2:51], 1, mean), main="SOC stock")
abline(0,1, col="red")
plot(pi, apply(temp3$SOC.d[,2:51], 1, mean), main="SOC density")
abline(0,1, col="red")

saveRDS(temp3, file=paste0("00_Files/Validation/Uncertainty_PICP_",year,".rds")) # save the PICP plots


### 3.2.2. Compile QCP plots ====
temp4 <-list() # container
temp4[[1]] <- matrix(NA, ncol=51, nrow=length(pi)); temp4[[1]][,1] = pi # matrix for SOC content
temp4[[2]] <- matrix(NA, ncol=51, nrow=length(pi)); temp4[[2]][,1] = pi # matrix for SOC stock
temp4[[3]] <- matrix(NA, ncol=51, nrow=length(pi)); temp4[[3]][,1] = pi # matrix for SOC density

index <- subset(1:99, 1:99 %% 2 == 0)

for(i in 1:50){ # SOC content
  
  for(j in 1:length(pi)){
    m <- index[j]
    temp4[[1]][j,(i+1)] <- sum(ifelse(temp1[[i]]$SOC.c[,m] >= temp2[[i]]$obs.SOC.c, yes=1, no=0))/nrow(temp1[[i]]$SOC.c)
    
  }
  
}

for(i in 1:50){ # SOC stock
  
  for(j in 1:length(pi)){
    m <- index[j]
    temp4[[2]][j,(i+1)] <- sum(ifelse(temp1[[i]]$SOC.s[,m] >= temp2[[i]]$obs.SOC.s, yes=1, no=0))/nrow(temp1[[i]]$SOC.s)
    
  }
  
}

for(i in 1:50){ # SOC density
  
  for(j in 1:length(pi)){
    m <- index[j]
    temp4[[3]][j,(i+1)] <- sum(ifelse(temp1[[i]]$SOC.d[,m] >= temp2[[i]]$obs.SOC.d, yes=1, no=0))/nrow(temp1[[i]]$SOC.d)
    
  }
  
}

temp4[[1]] <- data.frame(temp4[[1]]); names(temp4[[1]]) <- c("theo", paste0("fold.", 1:50))
temp4[[2]] <- data.frame(temp4[[2]]); names(temp4[[2]]) <- c("theo", paste0("fold.", 1:50))
temp4[[3]] <- data.frame(temp4[[3]]); names(temp4[[3]]) <- c("theo", paste0("fold.", 1:50))

names(temp4) <- c("SOC.c", "SOC.s", "SOC.d")

par(mfrow=c(1,3)) # quickly check the results
plot(pi, apply(temp4$SOC.c[,2:51], 1, mean), main="SOC content")
abline(0,1, col="red")
plot(pi, apply(temp4$SOC.s[,2:51], 1, mean), main="SOC stock")
abline(0,1, col="red")
plot(pi, apply(temp4$SOC.d[,2:51], 1, mean), main="SOC density")
abline(0,1, col="red")

saveRDS(temp4, file=paste0("00_Files/Validation/Uncertainty_QCP_",year,".rds")) # save the QCP plots
