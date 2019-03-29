## Report the Space Allocated for an Object
for (obj in ls()) { message(obj); print(object.size(get(obj)), units='auto') }

## downsample
train <- train %>% filter(booking_bool==0) %>%
  sample_n(sum(train$booking_bool==1), replace=F) %>%
  bind_rows(train %>% filter(booking_bool==1))

## Start H2O Cluster & Load Data
library(h2o)
h2o.init(nthreads=-1)
h2o.no_progress() ## Don't show progress bars in RMarkdown output

## Import a sample binary outcome train/test set into H2O
train <- as.h2o(train)

## Identify predictors and response
target <- c('click_bool', 'booking_bool')
x <- setdiff(names(train), target)

## For binary classification, response should be a factor
train[,target] <- as.factor(train[,target])

## Number of CV folds (to generate level-one data for stacking)
nfolds <- 5

## pre-vectorize prediction for two reponses
pred <- as.h2o((matrix(0, nrow(train), 2)))

## H2O base models
for (i in 1:2){
  
  print(sprintf("Training the Classifier for %s ...  ", target[i]))
  y <- target[i]
  
  # Tic 
  ptm <- proc.time()
  
  ## Train & Cross-validate a GBM
  my_gbm <- h2o.gbm(x=x,
                    y=y,
                    training_frame=train,
                    distribution="bernoulli",
                    max_depth=3,
                    min_rows=2,
                    learn_rate=0.2,
                    nfolds=nfolds,
                    fold_assignment="Modulo",
                    keep_cross_validation_predictions=TRUE,
                    seed=1)
  
  ## Train & Cross-validate a RF
  my_rf <- h2o.randomForest(x=x,
                            y=y,
                            training_frame=train,
                            nfolds=nfolds,
                            fold_assignment="Modulo",
                            keep_cross_validation_predictions=TRUE,
                            seed=1)
  
  ## Train & Cross-validate a LR
  my_lr <- h2o.glm(x=x,
                   y=y,
                   family='binomial', 
                   training_frame=train,
                   nfolds=nfolds,
                   fold_assignment="Modulo",
                   keep_cross_validation_predictions=TRUE)
  
  ## XGBoost base models
  
  ## Train & Cross-validate a (shallow) XGB-GBM
  my_xgb1 <- h2o.xgboost(x=x,
                         y=y,
                         training_frame=train,
                         distribution="bernoulli",
                         ntrees=50,
                         max_depth=3,
                         min_rows=2,
                         learn_rate=0.2,
                         nfolds=nfolds,
                         fold_assignment="Modulo",
                         keep_cross_validation_predictions=TRUE,
                         seed=1)
  
  ## Train & Cross-validate another (deeper) XGB-GBM
  my_xgb2 <- h2o.xgboost(x=x,
                         y=y,
                         training_frame=train,
                         distribution="bernoulli",
                         ntrees=50,
                         max_depth=8,
                         min_rows=1,
                         learn_rate=0.1,
                         sample_rate=0.7,
                         col_sample_rate=0.9,
                         nfolds=nfolds,
                         fold_assignment="Modulo",
                         keep_cross_validation_predictions=TRUE,
                         seed=1)
  
  ## Create a Stacked Ensemble
  ## Train a stacked ensemble using the H2O and XGBoost models from above
  base_models <- list(my_gbm@model_id, my_rf@model_id, my_lr@model_id,
                      my_xgb1@model_id, my_xgb2@model_id)
  
  ensemble <- h2o.stackedEnsemble(x=x,
                                  y=y,
                                  training_frame=train,
                                  base_models=base_models)
  
  print(sprintf("Time used: %s", (proc.time() - ptm)[3]))
  cat("\n")
  
  ## Eval ensemble performance on a test set
  perf <- h2o.performance(ensemble, newdata=train)
  pred[, i] <- predict(ensemble, newdata=train)[, 3]
  
  ## Compare to base learner performance on the test set
  get_auc <- function(mm) h2o.auc(h2o.performance(h2o.getModel(mm), newdata=train))
  baselearner_aucs <- sapply(base_models, get_auc)
  baselearner_best_auc_test <- max(baselearner_aucs)
  ensemble_auc_test <- h2o.auc(perf)
  
  ## report performance
  print("Evaluating the classifier:")
  print(sprintf("Best Base-learner Test AUC:  %s", baselearner_best_auc_test))
  print(sprintf("Ensemble Test AUC:  %s", ensemble_auc_test))
  cat("\n")
  
  print("Saving the classifier...")
  ptm = proc.time()
  if (i == 1){
    save(base_models, file = paths$model_path_click)
  } else {
    save(base_models, file = paths$model_path_book)
  }
  print(sprintf("Time used: %s", (proc.time() - ptm)[3]))
  cat("\n")
  
  cat("~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~\n")
  
}

## make prediction
test_result <- h2o.cbind(train$srch_id, train$prop_id, pred[, 1] + pred[, 2] * 4, train$click_bool, train$booking_bool)
names(test_result) <- c("SearchId","PropertyId", "Score", "click_bool", "booking_bool")
test_result <- as.data.frame(test_result)
row.names(test_result) <- NULL
test_result <- arrange(test_result, SearchId, -Score)
test_result$click_bool <- as.integer(test_result$click_bool) - 1
test_result$booking_bool <- as.integer(test_result$booking_bool) - 1