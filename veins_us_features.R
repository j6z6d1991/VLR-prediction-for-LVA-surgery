result_df<-read.csv("clinical_data.csv",header = T)
models <- c("LR", "LightGBM", "SVM", "RF", "CatBoost", "XGBoost")

metrics <- c(
  "AUC_train",
  "AUC_test",
  "Sensitivity",
  "Specificity",
  "PPV",
  "NPV",
  "F1-score",
  "Accuracy"
)

ML_performace <- data.frame(matrix(
  NA,
  nrow = length(metrics),
  ncol = length(models),
  dimnames = list(metrics, models)
))


# Define the ROC list
roclist_train <- list()
roclist_test <- list()


## 1. Data Preparation####
selected_cols <- c(14:18)
target_col <- 19

df_analysis <- result_df[, c(selected_cols, target_col)] %>%
  mutate(
    Bifurcation = ifelse(Bifurcation == "YES", 1, 0),
    Cortex = ifelse(Cortex == "Adipose layer", 1, 0),
    Compression_reflux = ifelse(Compression_reflux == "YES", 1, 0),
    Orthostatic_reflux = ifelse(Orthostatic_reflux == "YES", 1, 0),
    LVA_reflux = factor(LVA_reflux)
  ) %>%
  na.omit()

# Data splitting (70% for training, 30% for testing)
set.seed(2023)
train_idx <- sample(1:nrow(df_analysis), 0.7 * nrow(df_analysis))
train_data <- df_analysis[train_idx, ]
test_data <- df_analysis[-train_idx, ]

# Prepare the predictor variables and the outcome variable
x <- as.matrix(train_data[, 1:5])
y <- train_data[, 6]

## 2.logistic regression####
library(pROC)
### 2.1 building model####
lr_model <- glm(LVA_reflux ~ ., data = train_data, family = binomial())
summary(lr_model)

### 2.2 performance####
f0 <- glm(LVA_reflux ~ 1, data = train_data, family = binomial())
anova(f0, lr_model, test = "Chisq")

prob_train <- predict(object = lr_model,
                      newdata = train_data,
                      type = "response")
pred_train <- ifelse(prob_train >= 0.5, "YES", "NO")
pred_train <- factor(pred_train, levels = c("NO", "YES"), order = F)
roc_train <- roc(train_data[, 6], prob_train, ci = T)
auc_train <- auc(train_data[, 6], prob_train)
auc_ci_train <- ci.auc(roc_train)  
auc_train <- paste(
  round(auc_train, 3),
  "(",
  round(auc_ci_train[1], 3),
  "-",
  round(auc_ci_train[3], 3),
  ")",
  sep = ""
)
prob_test <- predict(object = lr_model,
                     newdata = test_data,
                     type = "response")
pred_test <- ifelse(prob_test >= 0.5, "YES", "NO")
pred_test <- factor(pred_test, levels = c("NO", "YES"), order = F)
roc_test <- roc(test_data[, 6], prob_test, ci = T)
auc_test <- auc(test_data[, 6], prob_test)
auc_ci_test <- ci.auc(roc_test)  
auc_test <- paste(round(auc_test, 3),
                  "(",
                  round(auc_ci_test[1], 3),
                  "-",
                  round(auc_ci_test[3], 3),
                  ")",
                  sep = "")
all <- rbind(train_data, test_data)
all_pred <- c(pred_train, pred_test)
f <- table(all$LVA_reflux, all_pred)
f
TP = f[2, 2]
FP = f[1, 2]
FN = f[2, 1]
TN = f[1, 1]
Sensitivity = TP / (TP + FN)
Specificity = TN / (TN + FP)
PPV = TP / (TP + FP)
NPV = TN / (TN + FN)
Recall = TP / (TP + FN)
F1_Score = 2 * (PPV * Sensitivity) / (PPV + Sensitivity)
Accuracy = (TP + TN) / (TP + FP + FN + TN)
ML_performace$LR <- c(
  auc_train,
  auc_test,
  round(Sensitivity, 3),
  round(Specificity, 3),
  round(PPV, 3),
  round(NPV, 3),
  round(F1_Score, 3),
  round(Accuracy, 3)
)

roclist_train[[1]] <- roc_train
roclist_test[[1]] <- roc_test

## 3.LightGBM####
library(tidymodels)
library(lightgbm)
library(bonsai)
### 3.2 building model####
set.seed(1)
lgb <- boost_tree() %>%
  set_engine("lightgbm") %>%
  set_mode("classification") %>%
  fit(LVA_reflux ~ ., data = train_data)

### 3.3 performace ####
library(DALEXtra)
lgb_exp <- explain_tidymodels(lgb,
                              data = train_data[, -6],
                              y = train_data$LVA_reflux,
                              label = "LightGBM")


prob_train <- predict(object = lgb_exp,
                      newdata = train_data,
                      type = "response")
pred_train <- ifelse(prob_train >= 0.5, "YES", "NO")
pred_train <- factor(pred_train, levels = c("NO", "YES"), order = F)
roc_train <- roc(train_data[, 6], prob_train, ci = T)
auc_train <- auc(train_data[, 6], prob_train)
auc_ci_train <- ci.auc(roc_train)  
auc_train <- paste(
  round(auc_train, 3),
  "(",
  round(auc_ci_train[1], 3),
  "-",
  round(auc_ci_train[3], 3),
  ")",
  sep = ""
)
prob_test <- predict(object = lgb_exp,
                     newdata = test_data,
                     type = "response")
pred_test <- ifelse(prob_test >= 0.5, "YES", "NO")
pred_test <- factor(pred_test, levels = c("NO", "YES"), order = F)
roc_test <- roc(test_data[, 6], prob_test, ci = T)
auc_test <- auc(test_data[, 6], prob_test)
auc_ci_test <- ci.auc(roc_test) 
auc_test <- paste(round(auc_test, 3),
                  "(",
                  round(auc_ci_test[1], 3),
                  "-",
                  round(auc_ci_test[3], 3),
                  ")",
                  sep = "")
all <- rbind(train_data, test_data)
all_pred <- c(pred_train, pred_test)
f <- table(all$LVA_reflux, all_pred)
f
TP = f[2, 2]
FP = f[1, 2]
FN = f[2, 1]
TN = f[1, 1]
Sensitivity = TP / (TP + FN)
Specificity = TN / (TN + FP)
PPV = TP / (TP + FP)
NPV = TN / (TN + FN)
Recall = TP / (TP + FN)
F1_Score = 2 * (PPV * Sensitivity) / (PPV + Sensitivity)
Accuracy = (TP + TN) / (TP + FP + FN + TN)
ML_performace$LightGBM <- c(
  auc_train,
  auc_test,
  round(Sensitivity, 3),
  round(Specificity, 3),
  round(PPV, 3),
  round(NPV, 3),
  round(F1_Score, 3),
  round(Accuracy, 3)
)
roclist_train[[2]] <- roc_train
roclist_test[[2]] <- roc_test
## --------------------4.SVM------------------------------------------------
library(e1071)
### 4.1 Hyperparameter tuning####
set.seed(123)
tune_model <- tune.svm(
  LVA_reflux ~ .,
  data = train_data,
  cost = 10^(-1:3),
  gamma = 10^(-3:1),
  tunecontrol = tune.control(sampling = "bootstrap", nboot = 100)#自助法
)
tune_model

### 4.2 building model####
svm_model <- svm(
  LVA_reflux ~ .,
  data = train_data,
  cost = 1000,
  gamma = 0.001,
  probability = T
)
print(svm_model)
summary(svm_model)
svm_model$index

### 4.3 performance####
prob_train <- predict(object = svm_model,
                      newdata = train_data,
                      probability = T)



prob_estimates <- attr(prob_train, "probabilities")
pred_train <- ifelse(prob_estimates[, 1] > 0.5, "YES", "NO")

roc_train <- roc(train_data[, 6], prob_estimates[, 1], ci = T)
auc_train <- auc(train_data[, 6], prob_estimates[, 1])
auc_ci_train <- ci.auc(roc_train)  
auc_train <- paste(
  round(auc_train, 3),
  "(",
  round(auc_ci_train[1], 3),
  "-",
  round(auc_ci_train[3], 3),
  ")",
  sep = ""
)
prob_test <- predict(object = svm_model,
                     newdata = test_data,
                     probability = T)
prob_estimates <- attr(prob_test, "probabilities")
pred_test <- ifelse(prob_estimates[, 1] > 0.5, "YES", "NO")

roc_test <- roc(test_data[, 6], prob_estimates[, 1], ci = T)
auc_test <- auc(test_data[, 6], prob_estimates[, 1])
auc_ci_test <- ci.auc(roc_test)  
auc_test <- paste(round(auc_test, 3),
                  "(",
                  round(auc_ci_test[1], 3),
                  "-",
                  round(auc_ci_test[3], 3),
                  ")",
                  sep = "")
all <- rbind(train_data, test_data)
all_pred <- c(pred_train, pred_test)
f <- table(all$LVA_reflux, all_pred)
f
TP = f[2, 2]
FP = f[1, 2]
FN = f[2, 1]
TN = f[1, 1]
Sensitivity = TP / (TP + FN)
Specificity = TN / (TN + FP)
PPV = TP / (TP + FP)
NPV = TN / (TN + FN)
Recall = TP / (TP + FN)
F1_Score = 2 * (PPV * Sensitivity) / (PPV + Sensitivity)
Accuracy = (TP + TN) / (TP + FP + FN + TN)
ML_performace$SVM <- c(
  auc_train,
  auc_test,
  round(Sensitivity, 3),
  round(Specificity, 3),
  round(PPV, 3),
  round(NPV, 3),
  round(F1_Score, 3),
  round(Accuracy, 3)
)
roclist_train[[3]] <- roc_train
roclist_test[[3]] <- roc_test
## --------------------5.random forest--------------------------------------
library(randomForest)
library(e1071)

### 5.1 Hyperparameter tuning####
tune_res <- tune.randomForest(
  x = x,
  y = y,
  nodesize = c(1:3),
  mtry = c(2, 4, 6, 8),
  ntree = 500
)
tune_res


### 5.2 building model####
set.seed(123)
rf_model <- randomForest(
  LVA_reflux ~ .,
  data = train_data,
  importance = T,
  mtry = 2,
  ntree = 500,
  nodesize = 3
)
rf_model

### 5.3 performance####
prob_train <- predict(object = rf_model,
                      newdata = train_data,
                      type = "prob")
pred_train <- ifelse(prob_train[, 2] > 0.5, "YES", "NO")

roc_train <- roc(train_data[, 6], prob_train[, 1], ci = T)
auc_train <- auc(train_data[, 6], prob_train[, 1])
auc_ci_train <- ci.auc(roc_train) 
auc_train <- paste(
  round(auc_train, 3),
  "(",
  round(auc_ci_train[1], 3),
  "-",
  round(auc_ci_train[3], 3),
  ")",
  sep = ""
)
prob_test <- predict(object = rf_model,
                     newdata = test_data,
                     type = "prob")
pred_test <- ifelse(prob_test[, 2] > 0.5, "YES", "NO")
roc_test <- roc(test_data[, 6], prob_test[, 1], ci = T)
auc_test <- auc(test_data[, 6], prob_test[, 1])
auc_ci_test <- ci.auc(roc_test)  
auc_test <- paste(round(auc_test, 3),
                  "(",
                  round(auc_ci_test[1], 3),
                  "-",
                  round(auc_ci_test[3], 3),
                  ")",
                  sep = "")
all <- rbind(train_data, test_data)
all_pred <- c(pred_train, pred_test)
f <- table(all$LVA_reflux, all_pred)
f
TP = f[2, 2]
FP = f[1, 2]
FN = f[2, 1]
TN = f[1, 1]
Sensitivity = TP / (TP + FN)
Specificity = TN / (TN + FP)
PPV = TP / (TP + FP)
NPV = TN / (TN + FN)
Recall = TP / (TP + FN)
F1_Score = 2 * (PPV * Sensitivity) / (PPV + Sensitivity)
Accuracy = (TP + TN) / (TP + FP + FN + TN)
ML_performace$RF <- c(
  auc_train,
  auc_test,
  round(Sensitivity, 3),
  round(Specificity, 3),
  round(PPV, 3),
  round(NPV, 3),
  round(F1_Score, 3),
  round(Accuracy, 3)
)
roclist_train[[4]] <- roc_train
roclist_test[[4]] <- roc_test


## --------------------6.catboost-------------------------------------------
library(catboost)

y <- ifelse(train_data$LVA_reflux == "YES", 1, 0)
train_pool <- catboost.load_pool(data = x, label = y)
train_pool

### 6.2 building model####
model <- catboost.train(
  train_pool,
  NULL,
  params = list(
    loss_function = 'Logloss',
    iterations = 100,

    metric_period = 10 # 
    #prediction_type=c("Class","Probability")
  )
)

model
catboost.get_feature_importance(model)
catboost.get_model_params(model)



### 6.2 performace####
prob_train <- catboost.predict(model, train_pool, prediction_type = "Probability")
pred_train <- ifelse(prob_train > 0.5, "YES", "NO")
roc_train <- roc(train_data[, 6], prob_train, ci = T)
auc_train <- auc(train_data[, 6], prob_train)
auc_ci_train <- ci.auc(roc_train)  
auc_train <- paste(
  round(auc_train, 3),
  "(",
  round(auc_ci_train[1], 3),
  "-",
  round(auc_ci_train[3], 3),
  ")",
  sep = ""
)

test_pool <- catboost.load_pool(test_data[, -6])
prob_test <- catboost.predict(model, test_pool, prediction_type = "Probability")
pred_test <- ifelse(prob_test > 0.5, "YES", "NO")
roc_test <- roc(test_data[, 6], prob_test, ci = T)
auc_test <- auc(test_data[, 6], prob_test)
auc_ci_test <- ci.auc(roc_test)  
auc_test <- paste(round(auc_test, 3),
                  "(",
                  round(auc_ci_test[1], 3),
                  "-",
                  round(auc_ci_test[3], 3),
                  ")",
                  sep = "")
all <- rbind(train_data, test_data)
all_pred <- c(pred_train, pred_test)
f <- table(all$LVA_reflux, all_pred)
f
TP = f[2, 2]
FP = f[1, 2]
FN = f[2, 1]
TN = f[1, 1]
Sensitivity = TP / (TP + FN)
Specificity = TN / (TN + FP)
PPV = TP / (TP + FP)
NPV = TN / (TN + FN)
Recall = TP / (TP + FN)
F1_Score = 2 * (PPV * Sensitivity) / (PPV + Sensitivity)
Accuracy = (TP + TN) / (TP + FP + FN + TN)
ML_performace$CatBoost <- c(
  auc_train,
  auc_test,
  round(Sensitivity, 3),
  round(Specificity, 3),
  round(PPV, 3),
  round(NPV, 3),
  round(F1_Score, 3),
  round(Accuracy, 3)
)
roclist_train[[5]] <- roc_train
roclist_test[[5]] <- roc_test


## --------------------7.xgboost--------------------------------------------
library(caret)
library(xgboost)
### 7.1Hyperparameter tuning####
grid <- expand.grid(
  nrounds = c(75, 100),
  colsample_bytree = 1,
  min_child_weight = 1,
  eta = c(0.01, 0.1, 0.3),
  gamma = c(0.5, 0.25),
  subsample = 0.5,
  max_depth = c(2, 3)
)

cntrl <- trainControl(
  method = "cv",
  number = 5,
  verboseIter = F,
  returnData = F,
  returnResamp = "final"
)

set.seed(1)
train.xgb <- train(
  x = train_data[, 1:5],
  y = train_data[, 5],
  trControl = cntrl,
  tuneGrid = grid,
  method = "xgbTree"
)

train.xgb
plot(train.xgb)
ggplot(train.xgb)

### 7.2 building model####
param <- list(
  objective = "binary:logistic",
  booster = "gbtree",
  eval_metric = "error",
  eta = 0.3,
  max_depth = 2,
  subsample = 0.5,
  colsample_bytree = 1,
  gamma = 0.25
)#


label <- ifelse(train_data[, 6] == "YES", 1, 0)
train.mat <- xgb.DMatrix(data = as.matrix(train_data[, c(1:5)]), label = label)
train.mat

set.seed(1)
xgb_model <- xgb.train(params = param,
                       data = train.mat,
                       nrounds = 100)

### 7.3 performance####
prob_train <- predict(object = xgb_model, newdata = train.mat)
pred_train <- ifelse(prob_train > 0.5, "YES", "NO")
roc_train <- roc(train_data[, 6], prob_train, ci = T)
auc_train <- auc(train_data[, 6], prob_train)
auc_ci_train <- ci.auc(roc_train)  
auc_train <- paste(
  round(auc_train, 3),
  "(",
  round(auc_ci_train[1], 3),
  "-",
  round(auc_ci_train[3], 3),
  ")",
  sep = ""
)

test.mat <- xgb.DMatrix(data = as.matrix(test_data[, c(1:5)]), label = test_data[, 6])
prob_test <- predict(object = xgb_model, newdata = test.mat)
pred_test <- ifelse(prob_test > 0.5, "YES", "NO")
roc_test <- roc(test_data[, 6], prob_test, ci = T)
auc_test <- auc(test_data[, 6], prob_test)
auc_ci_test <- ci.auc(roc_test)  
auc_test <- paste(round(auc_test, 3),
                  "(",
                  round(auc_ci_test[1], 3),
                  "-",
                  round(auc_ci_test[3], 3),
                  ")",
                  sep = "")
all <- rbind(train_data, test_data)
all_pred <- c(pred_train, pred_test)
f <- table(all$LVA_reflux, all_pred)
f
TP = f[2, 2]
FP = f[1, 2]
FN = f[2, 1]
TN = f[1, 1]
Sensitivity = TP / (TP + FN)
Specificity = TN / (TN + FP)
PPV = TP / (TP + FP)
NPV = TN / (TN + FN)
Recall = TP / (TP + FN)
F1_Score = 2 * (PPV * Sensitivity) / (PPV + Sensitivity)
Accuracy = (TP + TN) / (TP + FP + FN + TN)
ML_performace$XGBoost <- c(
  auc_train,
  auc_test,
  round(Sensitivity, 3),
  round(Specificity, 3),
  round(PPV, 3),
  round(NPV, 3),
  round(F1_Score, 3),
  round(Accuracy, 3)
)

roclist_train[[6]] <- roc_train
roclist_test[[6]] <- roc_test
names(roclist_train) <- models
names(roclist_test) <- models

write.csv(ML_performace, "ml_performance_us.csv")

## 8.visualization####
library(pROC)

pdf(file = "roc_train_us.pdf",
    width = 4.5,
    height = 4.5)
plot(
  roclist_train[[1]],
  col = "#FF2E63",
  legacy.axes = TRUE,

  print.auc = F,

  print.auc.y = 0.4,

  auc.polygon = TRUE,

  grid = c(0.2, 0.2),

  main = "Multi-gene ROC Analysis"
)

# Add additional ROC curves
colors <- c("#252A34",
            "#00DB00",
            "#FFA500",
            "#4B0082",
            "#0000FF",
            "#01106F")

for (j in 2:6) {
  plot(
    roclist_train[[j]],
    add = TRUE,
    col = colors[j],
    print.auc = F,
    print.auc.y = 0.4 - (j - 1) * 0.05
  ) 
}

# Add legend and statistical information
legend_text <- sapply(models, function(ml) {
  auc_val <- round(auc(roclist_train[[ml]]), 3)
  paste0(ml, " (AUC = ", auc_val, ")")
})

legend(
  "bottomright",
  legend = legend_text,
  col = c("#FF2E63", colors[2:length(models)]),
  lwd = 2,
  cex = 0.8,
  bty = "n"
)  

dev.off()

pdf(file = "roc_test_us.pdf",
    width = 4.5,
    height = 4.5)
plot(
  roclist_test[[1]],
  col = "#FF2E63",
  legacy.axes = TRUE,

  print.auc = F,

  print.auc.y = 0.4,

  auc.polygon = TRUE,

  grid = c(0.2, 0.2),

  main = "Multi-gene ROC Analysis"
)

# Add additional ROC curves
colors <- c("#252A34",
            "#00DB00",
            "#FFA500",
            "#4B0082",
            "#0000FF",
            "#01106F")

for (j in 2:6) {
  plot(
    roclist_test[[j]],
    add = TRUE,

    col = colors[j],
    print.auc = F,
    print.auc.y = 0.4 - (j - 1) * 0.05
  )  
}

# Add legend and statistical information
legend_text <- sapply(models, function(ml) {
  auc_val <- round(auc(roclist_test[[ml]]), 3)
  paste0(ml, " (AUC = ", auc_val, ")")
})

legend(
  "bottomright",
  legend = legend_text,
  col = c("#FF2E63", colors[2:length(models)]),
  lwd = 2,
  cex = 0.8,
  bty = "n"
)  

dev.off()
