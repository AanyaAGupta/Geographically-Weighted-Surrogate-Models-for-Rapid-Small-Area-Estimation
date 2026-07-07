# ============================================================
# XGBOOST model
# ============================================================
# NOTE:
# First, exclude obesity and smoking from VARS and run the script until the
# add-on section to generate obesity and smoking predictions.
#
# Then, include obesity and smoking in VARS and run the full script to
# generate the final chronic disease predictions.
# ============================================================

#load libraries
library(dplyr)
library(tidyr)
library(caret)
library(xgboost)

set.seed(123)

OUTCOME <- "per_copd"

#TRUE = only use counties present in BOTH training and transfer years
#FALSE = do not restrict to common FIPS
USE_COMMON_FIPS <- TRUE

#reading in data
train_year_path <- "./Data/2019_joined.csv"
test_year_path <- "./Data/2022_joined.csv"

#define variables you want to use
VARS <- c(
  "over_65",
  "white","female","bach","uninsured","poverty",
  "per_obesity","per_smoking","per_rural"
)

#helper methods to use throughout the code
std_fips <- function(x) sprintf("%05d", as.integer(gsub("\\D","", as.character(x))))
to_num   <- function(x) as.numeric(as.character(x))
rmse_fun <- function(p,o) sqrt(mean((p-o)^2))
r2_fun   <- function(p,o){ sse <- sum((o-p)^2); sst <- sum((o-mean(o))^2); 1 - sse/sst }
mae_fun  <- function(p,o) mean(abs(p-o))
r_fun    <- function(p,o) cor(p,o)

#function to load in data
load_joined_data <- function(file_path, OUTCOME, VARS) {
  
  df <- read.csv(file_path)
  
  df <- df %>%
    mutate(county_fips = std_fips(CountyFIPS)) %>%
    select(county_fips, all_of(c(OUTCOME, VARS)))
  
  cols_to_clean <- c(OUTCOME, VARS)
  
  #numeric
  df[cols_to_clean] <- lapply(df[cols_to_clean], to_num)
  
  #NA have been coded as -999
  df[cols_to_clean] <- lapply(df[cols_to_clean], function(x) {
    x[x == -999] <- NA
    return(x)
  })
  
  return(df)
}

#load training year data
df_train_year <- load_joined_data(train_year_path, OUTCOME, VARS)

#load transfer year data
df_test_year <- load_joined_data(test_year_path, OUTCOME, VARS)

#keep only counties present in BOTH training and transfer years
if (USE_COMMON_FIPS) {
  
  common_fips <- intersect(df_train_year$county_fips, df_test_year$county_fips)

  df_train_year <- df_train_year %>%
    filter(county_fips %in% common_fips) %>%
    arrange(county_fips)
  
  df_test_year <- df_test_year %>%
    filter(county_fips %in% common_fips) %>%
    arrange(county_fips)
  
}

#training year: drop rows missing outcome or predictors
df_train_year <- df_train_year %>% drop_na(all_of(c(OUTCOME, VARS)))

#save full transfer year before dropping rows
df_test_year_all <- df_test_year

#transfer year: predictors must be complete
df_test_year <- df_test_year %>% drop_na(all_of(VARS))

#80/20 split
idx <- createDataPartition(df_train_year[[OUTCOME]], p = 0.8, list = FALSE)
train_data <- df_train_year[idx, ]
test_data  <- df_train_year[-idx, ]

#matric for xgboost
dtrain <- xgb.DMatrix(data  = data.matrix(train_data[, VARS, drop = FALSE]), label = train_data[[OUTCOME]])
dholdout <- xgb.DMatrix(data = data.matrix(test_data[, VARS, drop = FALSE]))

# #define grid
# tgrid <- expand.grid(
#   nrounds = c(1200, 2000, 3000, 4000),
#   max_depth = c(5, 6, 7),
#   eta = c(0.03, 0.05, 0.08),
#   gamma = c(0, 1),
#   colsample_bytree = c(0.7, 0.85),
#   min_child_weight = c(1, 3, 5),
#   subsample = c(0.7, 0.85)
# )
#
# xgb_params <- list(score = Inf, nrounds = NA, params = NULL)
# for (i in seq_len(nrow(tgrid))) {
#   par <- tgrid[i, ]
#   params <- list(
#     objective = "reg:squarederror",
#     eval_metric = "rmse",
#     max_depth = par$max_depth,
#     eta = par$eta,
#     subsample = par$subsample,
#     colsample_bytree = par$colsample_bytree,
#     min_child_weight = par$min_child_weight,
#     gamma = par$gamma
#   )
#   cv <- xgb.cv(
#     params = params,
#     data = dtrain,
#     nrounds = par$nrounds,
#     nfold = 5,
#     early_stopping_rounds = 100,
#     verbose = 0
#   )
#   sc <- cv$evaluation_log$test_rmse_mean[cv$best_iteration]
#   if (!is.na(sc) && sc < xgb_params$score) {
#     xgb_params$score   <- sc
#     xgb_params$nrounds <- cv$best_iteration
#     xgb_params$params  <- params
#   }
# }
# 
# print(xgb_params$params)
# 
# cat("\nBest CV RMSE:", round(xgb_params$score, 4),
#     " @ nrounds:", xgb_params$nrounds, "\n")

xgb_params <- list(
  nrounds = 1000,
  params = list(
    objective = "reg:squarederror",
    eval_metric = "rmse",
    max_depth = 6,
    eta = 0.03,
    subsample = 0.7,
    colsample_bytree = 0.85,
    min_child_weight = 5,
    gamma = 0
  )
)

#train final model
fit_xgb <- xgb.train(params  = xgb_params$params, data = dtrain, nrounds = xgb_params$nrounds, verbose = 0)

#evaluate on model
pred_holdout <- predict(fit_xgb, dholdout)

cat("\n--- XGBOOST HOLDOUT (20%) ---\n")
cat("r   :", round(r_fun   (pred_holdout, test_data[[OUTCOME]]), 3), "\n")
cat("R2  :", round(r2_fun  (pred_holdout, test_data[[OUTCOME]]), 3), "\n")
cat("MAE :", round(mae_fun (pred_holdout, test_data[[OUTCOME]]), 3), "\n")
cat("RMSE:", round(rmse_fun(pred_holdout, test_data[[OUTCOME]]), 3), "\n")

#refit on training year data
dtrain_all <- xgb.DMatrix(data  = data.matrix(df_train_year[, VARS, drop = FALSE]), label = df_train_year[[OUTCOME]])
final_xgb <- xgb.train(params  = xgb_params$params, data = dtrain_all, nrounds = xgb_params$nrounds, verbose = 0)

#predict transfer year outcomes using global xgboost
dtest_year <- xgb.DMatrix(data = data.matrix(df_test_year[, VARS, drop = FALSE]))

pred_test_year <- predict(final_xgb, dtest_year)

out_test_year <- df_test_year %>%
  select(county_fips, any_of(OUTCOME)) %>% mutate(pred_outcome = pred_test_year)

eval_test_year <- out_test_year %>% drop_na(all_of(OUTCOME))

r_test    <- r_fun(eval_test_year$pred_outcome, eval_test_year[[OUTCOME]])
r2_test   <- r2_fun(eval_test_year$pred_outcome, eval_test_year[[OUTCOME]])
mae_test  <- mae_fun(eval_test_year$pred_outcome, eval_test_year[[OUTCOME]])
rmse_test <- rmse_fun(eval_test_year$pred_outcome, eval_test_year[[OUTCOME]])

# cat("\n--- GLOBAL XGBOOST: Train Year → Transfer Year ---\n")
# cat("r   :", round(r_test, 3), "\n")
# cat("R2  :", round(r2_test, 3), "\n")
# cat("MAE :", round(mae_test, 3), "\n")
# cat("RMSE:", round(rmse_test, 3), "\n")

# save transfer year predictions
#write.csv(out_test_year, paste0("./Scripts/XGBoost/global_xgboost_t2_", OUTCOME, ".csv"), row.names = FALSE)

# ============================================================
# ADD-ON: Update Transfer Year Predictions Using Predicted Obesity and Smoking Values
# ============================================================
#read predicted obesity and smoking from previous model outputs
pred_obesity <- read.csv("./Scripts/XGBoost/global_xgboost_t2_per_obesity.csv") %>%
  select(county_fips, pred_outcome) %>%
  mutate(
    county_fips = std_fips(county_fips),
    pred_outcome = to_num(pred_outcome)
  ) %>%
  rename(per_obesity = pred_outcome)

pred_smoking <- read.csv("./Scripts/XGBoost/global_xgboost_t2_per_smoking.csv") %>%
  select(county_fips, pred_outcome) %>%
  mutate(
    county_fips = std_fips(county_fips),
    pred_outcome = to_num(pred_outcome)
  ) %>%
  rename(per_smoking = pred_outcome)

#replace original transfer year obesity and smoking with predicted values
df_test_year_updated <- df_test_year_all %>%
  select(-per_obesity, -per_smoking) %>%
  left_join(pred_obesity, by = "county_fips") %>%
  left_join(pred_smoking, by = "county_fips") %>%
  drop_na(all_of(VARS))

#predict outcome using the same fixed model
dtest_year_updated <- xgb.DMatrix(data = data.matrix(df_test_year_updated[, VARS, drop = FALSE]))

pred_test_year_updated <- predict(final_xgb, dtest_year_updated)

out_test_year_updated <- df_test_year_updated %>%
  select(county_fips, any_of(OUTCOME)) %>%
  mutate(pred_outcome = pred_test_year_updated)

#save final predictions
# write.csv(
#   out_test_year_updated,
#   paste0("./Scripts/XGBoost/global_xgboost_t2_", OUTCOME, "_updated.csv"),
#   row.names = FALSE
# )

#updated evaluation
eval_test_year_updated <- out_test_year_updated %>% drop_na(all_of(OUTCOME))

cat("\n--- GLOBAL XGBOOST: Training Year → Transfer Year with Predicted Obesity + Smoking ---\n")
cat("r   :", round(r_fun(eval_test_year_updated$pred_outcome,
                         eval_test_year_updated[[OUTCOME]]), 3), "\n")
cat("R2  :", round(r2_fun(eval_test_year_updated$pred_outcome,
                          eval_test_year_updated[[OUTCOME]]), 3), "\n")
cat("MAE :", round(mae_fun(eval_test_year_updated$pred_outcome,
                           eval_test_year_updated[[OUTCOME]]), 3), "\n")
cat("RMSE:", round(rmse_fun(eval_test_year_updated$pred_outcome,
                            eval_test_year_updated[[OUTCOME]]), 3), "\n")