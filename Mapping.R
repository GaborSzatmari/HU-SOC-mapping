# Mapping soil organic carbon content, density, and stock for Hungary
# Using quantile regression forest
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
library(raster)
library(parallel)


# 2. Read the regression matrix with SOC data ----
points <- readRDS("00_Files/Points_Ready_to_use.rds")


# 3. Fine-tuning of mtry ----
set.seed(2023)
indicies <- CreateSpacetimeFolds(x= points,
                                 spacevar = "POINT",
                                 k = 10) # folds for leave-location-out cross-validation

t.control <- trainControl(method="cv",
                          index = indicies$index)

t.grid <- expand.grid(mtry=c(1:4, seq(5, 65, by=5)),
                      splitrule="variance",
                      min.node.size=c(5)) # set possible mtry values

SOC.ranger <- list() # container

set.seed(567)
SOC.ranger[[1]] <- train(x=points[,c(8:74)],
                         y=points$SOC.c,
                         method="ranger",
                         importance="impurity",
                         num.trees=500,
                         trControl=t.control,
                         tuneGrid=t.grid,
                         verbose=TRUE,
                         quantreg=TRUE,
                         num.threads=(detectCores()-1)) # fine-tune for SOC content

set.seed(567)
SOC.ranger[[2]] <- train(x=points[,c(8:74)],
                         y=points$SOC.s,
                         method="ranger",
                         importance="impurity",
                         num.trees=500,
                         trControl=t.control,
                         tuneGrid=t.grid,
                         verbose=TRUE,
                         quantreg=TRUE,
                         num.threads=(detectCores()-1)) # fine-tune for SOC stock

set.seed(567)
SOC.ranger[[3]] <- train(x=points[,c(8:74)],
                         y=points$SOC.d,
                         method="ranger",
                         importance="impurity",
                         num.trees=500,
                         trControl=t.control,
                         tuneGrid=t.grid,
                         verbose=TRUE,
                         quantreg=TRUE,
                         num.threads=(detectCores()-1)) # fine-tune for SOC density

names(SOC.ranger) <- c("SOC.c", "SOC.s", "SOC.d")

saveRDS(SOC.ranger, file="00_Files/SOC_ranger_models.rds") # save fitted QRF models


# 4. Mapping ----
year=1992 # set the year of interest


## 4.1. Load static environmental covariates ====
r <- list.files(path=paste0("00_Files/Covariates/static_covariates/"), pattern="*.tif$", full.names=TRUE)

r.static <- stack(r)


## 4.2. Load environmental covariates related to climate and land use ====
r.dynamic <- list.files(path=paste0("00_Files/Covariates/dynamic_covariates/"),
                        pattern=paste0(year, ".*\\.tif$"),
                        full.names=TRUE,
                        recursive = TRUE)[1:41]

r.dynamic <- stack(r.dynamic)

r <- stack(r.static, r.dynamic)


## 4.3. Load environmental covariates from remote sensing ====
r.satellite <- list.files(path=paste0("00_Files/Covariates/dynamic_covariates/04_RemoteSensing"),
                          pattern=paste0(year, ".*.tif$"),
                          full.names=TRUE)

r.satellite <- stack(r.satellite)

r <- stack(r, r.satellite)

r.df <- as.data.frame(r, xy=TRUE, na.rm=TRUE) # create data frame

r.df$Geology_02 <- as.factor(r.df$Geology_02) # factor type covariates
r.df$Soil_type <- as.factor(r.df$Soil_type)
r.df$Landcover <- as.factor(r.df$Landcover)


## 4.4. Prediction ====
SOC.names <- names(SOC.ranger)

for(i in 1:3){
  pred01 <- predict(SOC.ranger[[i]], r.df[1:2500000,], num.threads=detectCores()-1)
  pred02 <- predict(SOC.ranger[[i]], r.df[2500001:5000000,], num.threads=detectCores()-1)
  pred03 <- predict(SOC.ranger[[i]], r.df[5000001:7500000,], num.threads=detectCores()-1)
  pred04 <- predict(SOC.ranger[[i]], r.df[7500001:nrow(r.df),], num.threads=detectCores()-1)
  
  pred <- data.frame(x=r.df$x,
                     y=r.df$y,
                     ranger.pred=c(pred01, pred02, pred03, pred04),
                     year=year)
  saveRDS(pred, file=paste0("00_Files/Predictions/", SOC.names[i], "_pred_", year, ".rds"))
  
  rm(pred, pred01, pred02, pred03, pred04)
  gc()
  
}


## 4.5. Uncertainty ====
for(i in 1:3){
  pred01 <- predict(SOC.ranger[[i]]$finalModel, r.df[1:2500000,], num.threads=(detectCores()-1), type="quantiles", quantiles=c(0.05,0.95))
  pred01 <- data.frame(pred01$predictions)
  names(pred01) <- c("q05", "q95")
  
  pred02 <- predict(SOC.ranger[[i]]$finalModel, r.df[2500001:5000000,], num.threads=(detectCores()-1), type="quantiles", quantiles=c(0.05,0.95))
  pred02 <- data.frame(pred02$predictions)
  names(pred02) <- c("q05", "q95")
  
  pred03 <- predict(SOC.ranger[[i]]$finalModel, r.df[5000001:7500000,], num.threads=(detectCores()-1), type="quantiles", quantiles=c(0.05,0.95))
  pred03 <- data.frame(pred03$predictions)
  names(pred03) <- c("q05", "q95")
  
  pred04 <- predict(SOC.ranger[[i]]$finalModel, r.df[7500001:nrow(r.df),], num.threads=(detectCores()-1), type="quantiles", quantiles=c(0.05,0.95))
  pred04 <- data.frame(pred04$predictions)
  names(pred04) <- c("q05", "q95")
  
  pred <- data.frame(x=r.df$x,
                     y=r.df$y,
                     year=year)
  pred <- data.frame(pred, rbind(pred01, pred02, pred03, pred04))
  
  saveRDS(pred, file=paste0("00_Files/Predictions/", SOC.names[i], "_uncertainty_", year, ".rds"))
  
  rm(pred, pred01, pred02, pred03, pred04)
  gc()
  
}


# 5. Export GeoTIFFs ----
files <- list.files(path = "00_Files/Predictions/",
                    pattern = paste0("\\.rds$"),
                    full.names = TRUE) # read predictions and uncertainty quantifications

temp <- list()

for(i in 1:length(files)){
  temp[[i]] <- readRDS(files[i])
  temp[[i]] <- raster::rasterFromXYZ(temp[[i]])
}

temp <- stack(temp)

temp <- raster::dropLayer(temp, c(2,4,5,8,12,14,15,18,22,24,25,28)) # drop unnecessary layers

temp <- round(temp, digits = 2) # two-decimal precision

names(temp) <- c("SOC.c.pred.1992", "SOC.c.pred.2000",  "SOC.c.lower.1992", "SOC.c.upper.1992", "SOC.c.lower.2000", "SOC.c.upper.2000", "SOC.d.pred.1992", "SOC.d.pred.2000", "SOC.d.lower.1992", "SOC.d.upper.1992", "SOC.d.lower.2000", "SOC.d.upper.2000", "SOC.s.pred.1992", "SOC.s.pred.2000", "SOC.s.lower.1992", "SOC.s.upper.1992", "SOC.s.lower.2000", "SOC.s.upper.2000")

writeRaster(temp$SOC.c.pred.1992, filename="00_Files/Predictions/GeoTIFF/SOCc_0_30cm_1992_pred.tif", format="GTiff", overwrite=TRUE)
writeRaster(temp$SOC.c.lower.1992, filename="00_Files/Predictions/GeoTIFF/SOCc_0_30cm_1992_q05.tif", format="GTiff", overwrite=TRUE)
writeRaster(temp$SOC.c.upper.1992, filename="00_Files/Predictions/GeoTIFF/SOCc_0_30cm_1992_q95.tif", format="GTiff", overwrite=TRUE)
writeRaster(temp$SOC.c.pred.2000, filename="00_Files/Predictions/GeoTIFF/SOCc_0_30cm_2000_pred.tif", format="GTiff", overwrite=TRUE)
writeRaster(temp$SOC.c.lower.2000, filename="00_Files/Predictions/GeoTIFF/SOCc_0_30cm_2000_q05.tif", format="GTiff", overwrite=TRUE)
writeRaster(temp$SOC.c.upper.2000, filename="00_Files/Predictions/GeoTIFF/SOCc_0_30cm_2000_q95.tif", format="GTiff", overwrite=TRUE)

writeRaster(temp$SOC.s.pred.1992, filename="00_Files/Predictions/GeoTIFF/SOCs_0_30cm_1992_pred.tif", format="GTiff", overwrite=TRUE)
writeRaster(temp$SOC.s.lower.1992, filename="00_Files/Predictions/GeoTIFF/SOCs_0_30cm_1992_q05.tif", format="GTiff", overwrite=TRUE)
writeRaster(temp$SOC.s.upper.1992, filename="00_Files/Predictions/GeoTIFF/SOCs_0_30cm_1992_q95.tif", format="GTiff", overwrite=TRUE)
writeRaster(temp$SOC.s.pred.2000, filename="00_Files/Predictions/GeoTIFF/SOCs_0_30cm_2000_pred.tif", format="GTiff", overwrite=TRUE)
writeRaster(temp$SOC.s.lower.2000, filename="00_Files/Predictions/GeoTIFF/SOCs_0_30cm_2000_q05.tif", format="GTiff", overwrite=TRUE)
writeRaster(temp$SOC.s.upper.2000, filename="00_Files/Predictions/GeoTIFF/SOCs_0_30cm_2000_q95.tif", format="GTiff", overwrite=TRUE)

writeRaster(temp$SOC.d.pred.1992, filename="00_Files/Predictions/GeoTIFF/SOCd_0_30cm_1992_pred.tif", format="GTiff", overwrite=TRUE)
writeRaster(temp$SOC.d.lower.1992, filename="00_Files/Predictions/GeoTIFF/SOCd_0_30cm_1992_q05.tif", format="GTiff", overwrite=TRUE)
writeRaster(temp$SOC.d.upper.1992, filename="00_Files/Predictions/GeoTIFF/SOCd_0_30cm_1992_q95.tif", format="GTiff", overwrite=TRUE)
writeRaster(temp$SOC.d.pred.2000, filename="00_Files/Predictions/GeoTIFF/SOCd_0_30cm_2000_pred.tif", format="GTiff", overwrite=TRUE)
writeRaster(temp$SOC.d.lower.2000, filename="00_Files/Predictions/GeoTIFF/SOCd_0_30cm_2000_q05.tif", format="GTiff", overwrite=TRUE)
writeRaster(temp$SOC.d.upper.2000, filename="00_Files/Predictions/GeoTIFF/SOCd_0_30cm_2000_q95.tif", format="GTiff", overwrite=TRUE)
