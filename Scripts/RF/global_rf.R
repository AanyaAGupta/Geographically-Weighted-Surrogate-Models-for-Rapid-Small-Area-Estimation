# ============================================================
# GLOBAL RF
# ============================================================
# NOTE:
# First, exclude obesity and smoking from VARS and run the script until the
# add-on section to generate obesity and smoking predictions.
#
# Then, include obesity and smoking in VARS and run the full script to
# generate the final chronic disease predictions.
# ============================================================

#libraries
library(dplyr)
library(tidyr)
library(caret)
library(ranger)

set.seed(123)

OUTCOME <- "per_copd" #IMPORTANT: change based on outcome you want to predict

#TRUE = only use counties present in BOTH training and transfer years
#FALSE = do not restrict to common FIPS
USE_COMMON_FIPS <- TRUE

#reading in data
train_year_path <- "./Data/2019_joined.csv"
test_year_path  <- "./Data/2022_joined.csv"

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

#load training and transfer year data before dropping NAs
df_train_year <- load_joined_data(train_year_path, OUTCOME, VARS)
df_test_year  <- load_joined_data(test_year_path, OUTCOME, VARS)

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

#train rf
ctrl <- trainControl(method = "cv", number = 5)

p <- length(VARS)
grid_fast <- expand.grid(
  mtry = round(sqrt(p)),
  splitrule = "variance",
  min.node.size = 3
)

fit_rf <- train(
  x = train_data[, VARS, drop = FALSE],
  y = train_data[[OUTCOME]],
  method = "ranger",
  trControl = ctrl,
  tuneGrid = grid_fast,
  num.trees = 800,
  importance = "impurity",
  metric = "Rsquared"
)

#test holdout
pred_holdout <- predict(fit_rf, newdata = test_data[, VARS, drop = FALSE])

cat("\n--- GLOBAL RF HOLDOUT (20%) ---\n")
cat("r   :", round(r_fun   (pred_holdout, test_data[[OUTCOME]]), 3), "\n")
cat("R2  :", round(r2_fun  (pred_holdout, test_data[[OUTCOME]]), 3), "\n")
cat("MAE :", round(mae_fun (pred_holdout, test_data[[OUTCOME]]), 3), "\n")
cat("RMSE:", round(rmse_fun(pred_holdout, test_data[[OUTCOME]]), 3), "\n")

#refit training year data to model
fit_rf_final <- train(
  x = df_train_year[, VARS, drop = FALSE],
  y = df_train_year[[OUTCOME]],
  method = "ranger",
  trControl = trainControl(method = "none"),
  tuneGrid = grid_fast,
  num.trees = 800,
  importance = "impurity"
)

#predict transfer year outcomes using global RF trained on training year data
pred_test_year <- predict(fit_rf_final, newdata = df_test_year[, VARS, drop = FALSE])

out_test_year <- df_test_year %>%
  select(county_fips, any_of(OUTCOME)) %>%
  mutate(pred_outcome = pred_test_year)

#evaluate transfer year estimates
eval_test_year <- out_test_year %>% drop_na(all_of(OUTCOME))

r_test    <- r_fun(eval_test_year$pred_outcome, eval_test_year[[OUTCOME]])
r2_test   <- r2_fun(eval_test_year$pred_outcome, eval_test_year[[OUTCOME]])
mae_test  <- mae_fun(eval_test_year$pred_outcome, eval_test_year[[OUTCOME]])
rmse_test <- rmse_fun(eval_test_year$pred_outcome, eval_test_year[[OUTCOME]])

# cat("\n--- GLOBAL RF: Training Year → Transfer Year ---\n")
# cat("r   :", round(r_test, 3), "\n")
# cat("R2  :", round(r2_test, 3), "\n")
# cat("MAE :", round(mae_test, 3), "\n")
# cat("RMSE:", round(rmse_test, 3), "\n")

#save transfer year predictions
#write.csv(out_test_year, paste0("./Scripts/RF/global_rf_t2_", OUTCOME, ".csv"), row.names = FALSE)

# ============================================================
# ADD-ON: Update Transfer Year Predictions Using Predicted Obesity and Smoking Values
# ============================================================
#read predicted obesity and smoking from previous model outputs
pred_obesity <- read.csv("./Scripts/RF/global_rf_t2_per_obesity.csv") %>%
  select(county_fips, pred_outcome) %>%
  mutate(
    county_fips = std_fips(county_fips),
    pred_outcome = to_num(pred_outcome)
  ) %>%
  rename(per_obesity = pred_outcome)

pred_smoking <- read.csv("./Scripts/RF/global_rf_t2_per_smoking.csv") %>%
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
pred_test_year_updated <- predict(
  fit_rf_final,
  newdata = df_test_year_updated[, VARS, drop = FALSE]
)

out_test_year_updated <- df_test_year_updated %>%
  select(county_fips, any_of(OUTCOME)) %>%
  mutate(pred_outcome = pred_test_year_updated)

#save final predictions
# write.csv(
#   out_test_year_updated,
#   paste0("./Scripts/RF/global_rf_t2_", OUTCOME, "_updated.csv"),
#   row.names = FALSE
# )

#updated evaluation
eval_test_year_updated <- out_test_year_updated %>% drop_na(all_of(OUTCOME))

cat("\n--- GLOBAL RF: Training Year → Transfer Year with Predicted Obesity + Smoking ---\n")
cat("r   :", round(r_fun(eval_test_year_updated$pred_outcome,
                         eval_test_year_updated[[OUTCOME]]), 3), "\n")
cat("R2  :", round(r2_fun(eval_test_year_updated$pred_outcome,
                          eval_test_year_updated[[OUTCOME]]), 3), "\n")
cat("MAE :", round(mae_fun(eval_test_year_updated$pred_outcome,
                           eval_test_year_updated[[OUTCOME]]), 3), "\n")
cat("RMSE:", round(rmse_fun(eval_test_year_updated$pred_outcome,
                            eval_test_year_updated[[OUTCOME]]), 3), "\n")