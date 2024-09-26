# Installing Libraries
install.packages("reticulate")
reticulate::py_install("optuna")

# Importing packages
library(reticulate)
library(lightgbm)
library(tidymodels)
library(tidyverse)
library(mlbench)
library(bonsai)
library(yardstick)
optuna <- import('optuna')

# Loading Dataset from mlbench package
data(PimaIndiansDiabetes)
diab <- PimaIndiansDiabetes

# Printing missing values
colSums(is.na(diab))

# Overview of the dataset
glimpse(diab)

# Counting zeroes in the dataset
zero_cnt <- diab %>%
  summarise(across(everything(), ~ sum(. == 0, na.rm = TRUE)))
print(zero_cnt)

# Replacing the zeroes with NA
new_df <- diab %>%
  mutate(across(c(pregnant, glucose, pressure, insulin, mass, triceps), 
                ~if_else(. == 0, as.numeric(NA), .)))

# Checking again for the zeroes
zero_cnt <- new_df %>%
  summarise(across(everything(), ~ sum(. == 0, na.rm = TRUE)))
print(zero_cnt)

# Splitting the dataset in 80:20 ratio 
set.seed(123)
df <- initial_split(new_df, prop = 4/5)
df_train <- training(df)
df_test <- testing(df)

# Printing the splitted dataset
cat("Total data count:", nrow(new_df), "\n")
cat("Train data count:", nrow(df_train), "\n")
cat("Test data count:", nrow(df_test), "\n")

# Creating recipe for data preprocessing
df_recipe <- recipe(diabetes ~ pregnant + glucose + pressure + triceps + insulin + mass + pedigree + age, 
                          data = new_df) %>%
  step_normalize(all_numeric_predictors()) %>%
  step_impute_knn(all_predictors())

#-------------------MODEL BUILDING-----------------#

#--------------------------------------------------#
#------------------BASELINE MODEL------------------#
#--------------------------------------------------#
# Baseline model
base_model <- boost_tree() %>%
  set_engine("lightgbm") %>%
  set_mode("classification")

# Workflow 
base_wflow <- workflow() %>%
  add_recipe(df_recipe) %>%
  add_model(base_model)

# Fitting model
base_fit <- base_wflow %>%
  fit(data = df_train)

# Evaluating model
base_pred <- base_fit %>%
  predict(new_data = df_test) %>%
  bind_cols(df_test)

# Baseline model accuracy
base_mt <- base_pred %>%
  metrics(truth = diabetes, estimate = .pred_class)
base_acc <- base_mt %>%
  filter(.metric == "accuracy") %>%
  pull(.estimate) * 100
cat(sprintf("The accuracy of baseline model is: %.2f%%\n", base_acc))

#-----------------------------------------------------#
#------------------GRID SEARCH MODEL------------------#
#-----------------------------------------------------#
# Cross-validation fold
df_cv <- vfold_cv(df_train, v = 5)

# Grid search model
grid_model <- boost_tree(
  trees = tune(),
  learn_rate = tune(),
  tree_depth = tune()
) %>%
  set_engine("lightgbm") %>%
  set_mode("classification")

# Workflow
grid_wf <- workflow() %>%
  add_recipe(df_recipe) %>%
  add_model(grid_model)

# Defining grid 
grid <- grid_regular(
  parameters(grid_model),
  levels = 3
)

# Performing grid search
set.seed(42)
grid_res <- grid_wf %>%
  tune_grid(resamples = df_cv, grid = grid, metrics = metric_set(accuracy))

# Selecting best model based on accuracy
best_grid <- grid_res %>%
  select_best(metric = "accuracy")

# Finalizing the workflow with the best parameters
final_grid_wf <- grid_wf %>%
  finalize_workflow(best_grid)

# Fitting model
final_grid_fit <- final_grid_wf %>%
  fit(data = df_train)

# Evaluating model
grid_pred <- final_grid_fit %>%
  predict(new_data = df_test) %>%
  bind_cols(df_test)

# Grid search model accuracy
grid_mt <- grid_pred %>%
  metrics(truth = diabetes, estimate = .pred_class)
grid_acc <- grid_mt %>%
  filter(.metric == "accuracy") %>%
  pull(.estimate) * 100
cat(sprintf("The accuracy of grid search model is: %.2f%%\n",grid_acc))

#------------------------------------------------------#
#------------------OPTUNA TUNED MODEL------------------#
#------------------------------------------------------#
# Defining objective function 
objective_lgbm <- function(trial) {
  
  # hyperparameter search space 
  trees <- trial$suggest_int("trees", 2000, 2500)
  learning_rate <- trial$suggest_loguniform("learning_rate", 0.001, 0.1)
  tree_depth <- trial$suggest_int("tree_depth", 3, 15)  

  # Tuned model
  lgbm_td <- boost_tree(trees = trees, 
                           learn_rate = learning_rate, 
                           tree_depth = tree_depth) %>%
    set_engine("lightgbm") %>%
    set_mode("classification")
  
  # Workflow 
  lgbm_wf_td <- workflow() %>%
    add_recipe(df_recipe) %>%
    add_model(lgbm_td)
  
  # Cross-validation to evaluate the model
  cv_res <- lgbm_wf_td %>%
    fit_resamples(resamples = df_cv, 
                  metrics = metric_set(accuracy),
                  control = control_resamples(save_pred = TRUE))
  mean_acc <- collect_metrics(cv_res) %>%
    filter(.metric == "accuracy") %>%
    summarize(mean_acc = mean(mean))
  
  return(-mean_acc$mean_acc)  
}
op_study <- optuna$create_study(direction = "minimize")
op_study$optimize(objective_lgbm, n_trials = 50)

# Getting the best hyperparameters 
best_params <- op_study$best_params
print(best_params)

# Recreating the tuned model with best hyperparameters
final <- boost_tree(
  trees = best_params$trees, 
  learn_rate = best_params$learning_rate, 
  tree_depth = best_params$tree_depth,
) %>%
  set_engine("lightgbm") %>%
  set_mode("classification")

# Workflow 
td_wf <- workflow() %>%
  add_recipe(df_recipe) %>%
  add_model(final)

# Fitting model
td_fit <- td_wf %>%
  fit(data = df_train)

# Evaluating model
td_pred <- td_fit %>%
  predict(new_data = df_test) %>%
  bind_cols(df_test)

# Tuned model accuracy
td_mt <- td_pred %>%
  metrics(truth = diabetes, estimate = .pred_class)
tun_acc <- td_mt %>%
  filter(.metric == "accuracy") %>%
  pull(.estimate) * 100
cat(sprintf("The accuracy of optuna tuned model is: %.2f%%\n",tun_acc))

#------------------------------------------------------#
#------------------CONFUSION MATRIX--------------------#
#------------------------------------------------------#

# Baseline model confusion matrix
base_pred %>%
  conf_mat(truth = diabetes, estimate = .pred_class) %>%
  pluck(1) %>% as_tibble() %>%
  ggplot(aes(Truth, Prediction, alpha = n)) +
  geom_tile(fill = "#3a86ff", show.legend = FALSE) +
  geom_text(alpha = 0.8, size = 5,aes(label = n), colour = "black")+
 labs(title = "Baseline Model Confusion Matrix")+theme_bw()+
 theme(plot.title = element_text(size = 18, hjust = 0.5))

# Grid search model confusion matrix
grid_pred %>%
  conf_mat(truth = diabetes, estimate = .pred_class) %>%
  pluck(1) %>% as_tibble() %>%
  ggplot(aes(Truth, Prediction, alpha = n)) +
  geom_tile(fill = "#3a86ff", show.legend = FALSE) +
  geom_text(alpha = 0.8, size = 5,aes(label = n), colour = "black")+
 labs(title = "Grid Search Model Confusion Matrix")+theme_bw()+
 theme(plot.title = element_text(size = 18, hjust = 0.5))

# Tuned model confusion matrix
td_pred %>%
  conf_mat(truth = diabetes, estimate = .pred_class) %>%
  pluck(1) %>% as_tibble() %>%
  ggplot(aes(Truth, Prediction, alpha = n)) +
  geom_tile(fill = "#3a86ff", show.legend = FALSE) +
  geom_text(alpha = 0.8, size = 5,aes(label = n), colour = "black")+
 labs(title = "Optuna Tuned Model Confusion Matrix")+theme_bw()+
 theme(plot.title = element_text(size = 18, hjust = 0.5))