# ============================================================
# GWRF model
# ============================================================
# NOTE:
# First, exclude obesity and smoking from VARS and run the script until the
# add-on section to generate obesity and smoking predictions.
#
# Then, include obesity and smoking in VARS and run the full script to
# generate the final chronic disease predictions.
# ============================================================

#load libraries
library(SpatialML)
library(sf)
library(dplyr)
library(caret)
library(stringr)
library(parallel)
library(tidyr)

set.seed(123)


OUTCOME <- "per_copd" #change for the health outcome you came to generate
CONUS   <- TRUE

#load in data
train_year_path <- "./Data/2019ACS_2023cdcPlaces_2019CHR.shp"
test_year_path <- "./Data/2022ACS_2024cdcPlaces_2022CHR.shp"

#define variables
VARS <- c(
  "over_65",
  "white","female","bach","uninsured","poverty",
  "per_obesit","per_smokin","per_rural"
)

CRS_EPSG    <- 5070
N_THREADS   <- max(1, detectCores() - 1)
NTREE_FINAL <- 400
KERNEL      <- "adaptive"
IMP         <- "none"

#helper methods (similar to rf model)
pad5   <- function(x) stringr::str_pad(as.character(x), 5, pad = "0")
to_num <- function(x) as.numeric(as.character(x))
rmse   <- function(p,o) sqrt(mean((p-o)^2))
r2     <- function(p,o){ sse <- sum((o-p)^2); sst <- sum((o-mean(o))^2); 1 - sse/sst }
mae    <- function(p,o) mean(abs(p-o))
r_fun  <- function(p,o) cor(p,o)

prep_year <- function(shp_path, fips_col, outcome_col, vars_keep, conus = FALSE) {
  joined <- st_read(shp_path, quiet = TRUE) %>%
    rename(county_fips = all_of(fips_col)) %>%
    mutate(county_fips = pad5(county_fips))
  
  if (conus && "StateAbbr" %in% names(joined)) {
    joined <- joined %>% filter(!StateAbbr %in% c("AK", "HI", "PR"))
  }
  
  vars_final <- intersect(vars_keep, names(joined))
  
  # numeric-only
  for (v in vars_final) joined[[v]] <- to_num(joined[[v]])
  if (outcome_col %in% names(joined)) joined[[outcome_col]] <- to_num(joined[[outcome_col]])
  
  # NA have been coded as -999
  joined <- joined %>% mutate(across(all_of(vars_final), ~ replace(., . == -999, NA)))
  
  if (outcome_col %in% names(joined)) {
    joined <- joined %>%
      mutate(across(all_of(outcome_col), ~ replace(., . == -999, NA)))
  }
  
  # project + centroids
  joined_5070 <- st_transform(joined, CRS_EPSG)
  pts         <- st_centroid(joined_5070)
  coords      <- st_coordinates(pts)
  
  df <- joined_5070 %>% st_drop_geometry() %>%
    select(any_of(c("county_fips", outcome_col, vars_final))) %>%
    as.data.frame()
  df$.rid <- seq_len(nrow(df))
  list(df = df, coords = coords, vars = vars_final) 
}

cat("== Prep Training Year...\n")
train_year <- prep_year(train_year_path, "CountyFIPS", OUTCOME, VARS, conus = CONUS)

cat("== Prep Transfer Year...\n")
test_year <- prep_year(test_year_path, "CountyFIPS", OUTCOME, VARS, conus = CONUS)

# keep only counties present in BOTH training and transfer years
common_fips <- intersect(train_year$df$county_fips, test_year$df$county_fips)

train_year$df <- train_year$df %>%
  filter(county_fips %in% common_fips) %>%
  arrange(county_fips)

test_year$df <- test_year$df %>%
  filter(county_fips %in% common_fips) %>%
  arrange(county_fips)

train_df <- train_year$df %>% filter(is.finite(.data[[OUTCOME]]))

# drop near-zero variance cols
nzv <- nearZeroVar(train_df[, train_year$vars, drop = FALSE])
pred_vars <- if (length(nzv)) train_year$vars[-nzv] else train_year$vars
stopifnot(length(pred_vars) > 1)

train_df <- train_df %>% filter(if_all(all_of(pred_vars), ~ is.finite(.)))

train_coords <- train_year$coords[train_df$.rid, , drop = FALSE]

# 80/20 holdout
idx <- createDataPartition(train_df[[OUTCOME]], p = 0.80, list = FALSE)
df_tr <- train_df[idx, c(OUTCOME, pred_vars, ".rid", "county_fips"), drop = FALSE] 
df_te <- train_df[-idx, c(OUTCOME, pred_vars, ".rid", "county_fips"), drop = FALSE]
C_tr  <- train_coords[idx, , drop = FALSE]
C_te  <- train_coords[-idx, , drop = FALSE]
class(df_tr) <- "data.frame"; class(df_te) <- "data.frame"

p        <- length(pred_vars)
mtry_use <- max(1, floor(sqrt(p)))            
bw_use   <- min(100, nrow(df_tr) - 1)          

form <- as.formula(paste(OUTCOME, "~", paste(pred_vars, collapse = " + ")))

cat(sprintf("== Train GWRF: n=%d, p=%d, bw=%d, mtry=%d, ntree=%d, threads=%d\n",
            nrow(df_tr), p, bw_use, mtry_use, NTREE_FINAL, N_THREADS))

gwrf_model <- SpatialML::grf(
  formula       = form,
  dframe        = df_tr[, c(OUTCOME, pred_vars), drop = FALSE],
  coords        = C_tr,
  bw            = bw_use,
  kernel        = KERNEL,
  ntree         = NTREE_FINAL,
  nthreads      = N_THREADS,
  mtry          = mtry_use,
  forests       = TRUE,
  write.forest  = TRUE,
  importance    = "impurity",
  geo.weighted  = TRUE,
  print.results = TRUE
)

#training year holdout
pred_train <- SpatialML::predict.grf(
  gwrf_model,
  new.data   = as.data.frame(cbind(df_te[, pred_vars, drop = FALSE], X = C_te[,1], Y = C_te[,2])),
  x.var.name = "X",
  y.var.name = "Y",
  nthreads   = N_THREADS
)

cat("\n--- GWRF TRAINING YEAR HOLDOUT (", OUTCOME, ") ---\n", sep = "")
cat("r   :", round(r_fun(pred_train, df_te[[OUTCOME]]), 3), "\n")
cat("R2  :", round(r2   (pred_train, df_te[[OUTCOME]]), 3), "\n")
cat("MAE :", round(mae  (pred_train, df_te[[OUTCOME]]), 3), "\n")
cat("RMSE:", round(rmse (pred_train, df_te[[OUTCOME]]), 3), "\n")

# refit on all training year data
df_all <- train_df[, c(OUTCOME, pred_vars, ".rid", "county_fips"), drop = FALSE]
C_all  <- train_coords
class(df_all) <- "data.frame"

gwrf_final <- SpatialML::grf(
  formula       = form,
  dframe        = df_all[, c(OUTCOME, pred_vars), drop = FALSE],
  coords        = C_all,
  bw            = bw_use,
  kernel        = KERNEL,
  ntree         = NTREE_FINAL,
  nthreads      = N_THREADS,
  mtry          = mtry_use,
  forests       = TRUE,
  write.forest  = TRUE,
  importance    = "impurity",
  geo.weighted  = TRUE,
  print.results = TRUE
)

# transfer year predictions
pred_keep <- intersect(pred_vars, names(test_year$df))
df_test_year <- test_year$df %>% filter(if_all(all_of(pred_keep), ~ is.finite(.)))
coords_test_year <- test_year$coords[df_test_year$.rid, , drop = FALSE]

cat(sprintf("== Predict Transfer Year: n=%d\n", nrow(df_test_year)))
pred_testyr <- SpatialML::predict.grf(
  gwrf_final,
  new.data   = as.data.frame(cbind(df_test_year[, pred_keep, drop = FALSE], X = coords_test_year[,1], Y = coords_test_year[,2])),
  x.var.name = "X",
  y.var.name = "Y",
  nthreads   = N_THREADS
)

out_test_year <- df_test_year %>%
  select(county_fips, any_of(OUTCOME)) %>%  mutate(pred_outcome = pred_testyr)

eval_test_year <- out_test_year %>% drop_na(all_of(OUTCOME))

r_test    <- r_fun(eval_test_year$pred_outcome, eval_test_year[[OUTCOME]])
r2_test   <- r2(eval_test_year$pred_outcome, eval_test_year[[OUTCOME]])
mae_test  <- mae(eval_test_year$pred_outcome, eval_test_year[[OUTCOME]])
rmse_test <- rmse(eval_test_year$pred_outcome, eval_test_year[[OUTCOME]])

# cat("\n--- GWRF: Training Year → Transfer Year ---\n")
# cat("r   :", round(r_test, 3), "\n")
# cat("R2  :", round(r2_test, 3), "\n")
# cat("MAE :", round(mae_test, 3), "\n")
# cat("RMSE:", round(rmse_test, 3), "\n")

#save transfer year predictions
#write.csv(out_test_year, paste0("./Scripts/GWRF/gwrf_t2_", OUTCOME, ".csv"), row.names = FALSE)

# ============================================================
# ADD-ON: Update Transfer Year Predictions Using Predicted Obesity and Smoking Values
# ============================================================
#read predicted obesity and smoking from previous model outputs
pred_obesity <- read.csv("./Scripts/GWRF/gwrf_t2_per_obesit.csv") %>%
  select(county_fips, pred_outcome) %>%
  mutate(
    county_fips = pad5(county_fips),
    pred_outcome = to_num(pred_outcome)
  ) %>%
  rename(per_obesit = pred_outcome)

pred_smoking <- read.csv("./Scripts/GWRF/gwrf_t2_per_smokin.csv") %>%
  select(county_fips, pred_outcome) %>%
  mutate(
    county_fips = pad5(county_fips),
    pred_outcome = to_num(pred_outcome)
  ) %>%
  rename(per_smokin = pred_outcome)

#replace original transfer year obesity and smoking with predicted values
df_test_year_updated <- test_year$df %>%
  select(-per_obesit, -per_smokin) %>%
  left_join(pred_obesity, by = "county_fips") %>%
  left_join(pred_smoking, by = "county_fips") %>%
  drop_na(all_of(pred_keep))

coords_teyr_updated <- test_year$coords[df_test_year_updated$.rid, , drop = FALSE]

pred_testyr_updated <- SpatialML::predict.grf(
  gwrf_final,
  new.data   = as.data.frame(cbind(df_test_year_updated[, pred_keep, drop = FALSE], X = coords_teyr_updated[,1], Y = coords_teyr_updated[,2])),
  x.var.name = "X",
  y.var.name = "Y",
  nthreads   = N_THREADS
)

out_test_year_updated <- df_test_year_updated %>%
  select(county_fips, any_of(OUTCOME)) %>%
  mutate(pred_outcome = pred_testyr_updated)

#save final predictions
# write.csv(
#   out_test_year_updated,
#   paste0("./Scripts/GWRF/gwrf_t2_", OUTCOME, "_updated.csv"),
#   row.names = FALSE
# )

#updated evaluation
eval_test_year_updated <- out_test_year_updated %>% drop_na(all_of(OUTCOME))

cat("\n--- GWRF: Training Year → Transfer Year with Predicted Obesity + Smoking ---\n")
cat("r   :", round(r_fun(eval_test_year_updated$pred_outcome,
                         eval_test_year_updated[[OUTCOME]]), 3), "\n")
cat("R2  :", round(r2(eval_test_year_updated$pred_outcome,
                      eval_test_year_updated[[OUTCOME]]), 3), "\n")
cat("MAE :", round(mae(eval_test_year_updated$pred_outcome,
                       eval_test_year_updated[[OUTCOME]]), 3), "\n")
cat("RMSE:", round(rmse(eval_test_year_updated$pred_outcome,
                        eval_test_year_updated[[OUTCOME]]), 3), "\n")