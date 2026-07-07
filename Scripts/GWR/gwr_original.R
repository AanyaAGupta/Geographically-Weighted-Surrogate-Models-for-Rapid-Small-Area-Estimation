# ============================================================
# GWR
# ============================================================
# NOTE:
# First, exclude obesity and smoking from VARS and run the script until the
# add-on section to generate obesity and smoking predictions.
#
# Then, include obesity and smoking in VARS and run the full script to
# generate the final chronic disease predictions.
# ============================================================

library(GWmodel)
library(sf)
library(dplyr)
library(sp)
library(stringr)
library(tidyr)

set.seed(123)

OUTCOME  <- "per_copd"
CONUS    <- TRUE
CRS_EPSG <- 5070
KNN      <- 200 

# ---- FILES ----
train_year_path <- "./Data/2019ACS_2023cdcPlaces_2019CHR.shp"
test_year_path  <- "./Data/2022ACS_2024cdcPlaces_2022CHR.shp"

VARS <- c(
  "over_65","white","female","bach","uninsured","poverty",
  "per_smokin","per_obesit","per_rural"
)

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
  
  for (v in vars_final) joined[[v]] <- to_num(joined[[v]])
  if (outcome_col %in% names(joined)) joined[[outcome_col]] <- to_num(joined[[outcome_col]])
  
  joined <- joined %>% mutate(across(all_of(vars_final), ~ replace(., . == -999, NA)))
  
  if (outcome_col %in% names(joined)) {
    joined <- joined %>%
      mutate(across(all_of(outcome_col), ~ replace(., . == -999, NA)))
  }
  
  # project + centroids
  joined_5070 <- st_transform(joined, CRS_EPSG)
  pts         <- st_centroid(joined_5070)
  coords      <- st_coordinates(pts)
  crs_wkt     <- st_crs(joined_5070)$wkt
  
  df <- joined_5070 %>% st_drop_geometry() %>%
    select(any_of(c("county_fips", outcome_col, vars_final))) %>%
    as.data.frame()
  df$.rid <- seq_len(nrow(df))
  list(df = df, coords = coords, vars = vars_final, crs_wkt = crs_wkt) 
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

COMMON_VARS <- intersect(train_year$vars, test_year$vars)

train_df <- train_year$df %>%
  filter(
    if_all(all_of(COMMON_VARS), is.finite),
    is.finite(.data[[OUTCOME]])
  )
coords_train <- train_year$coords[train_df$.rid, , drop = FALSE]

test_df <- test_year$df %>%
  filter(if_all(all_of(COMMON_VARS), is.finite))
coords_test <- test_year$coords[test_df$.rid, , drop = FALSE]

crs_obj <- CRS(train_year$crs_wkt)

sp_train <- SpatialPointsDataFrame(
  coords      = coords_train,
  data        = train_df[, c("county_fips", OUTCOME, COMMON_VARS), drop = FALSE],
  proj4string = crs_obj
)

sp_test <- SpatialPointsDataFrame(
  coords      = coords_test,
  data        = test_df[, c("county_fips", OUTCOME, COMMON_VARS), drop = FALSE],
  proj4string = crs_obj
)

form  <- as.formula(paste(OUTCOME, "~", paste(COMMON_VARS, collapse = " + ")))
k_use <- min(KNN, nrow(sp_train) - 1L)

cat(sprintf("== GWR (training year → predict transfer year): n_train=%d, p=%d, k=%d\n",
            nrow(sp_train), length(COMMON_VARS), k_use))

gwr_pred <- GWmodel::gwr.predict(
  formula     = form,
  data        = sp_train,  
  predictdata = sp_test,  
  bw          = k_use,
  kernel      = "bisquare",
  adaptive    = TRUE,
  longlat     = FALSE
)

SDF <- as.data.frame(gwr_pred$SDF)

#using the coefficients
# coef_names <- paste0(c("Intercept", COMMON_VARS), "_coef")
# coef_table <- data.frame(
#    county_fips = test_df$county_fips,
#    X           = coords_test[,1],
#    Y           = coords_test[,2],
#    SDF[, coef_names, drop = FALSE]
# )

pred_testyr <- as.numeric(SDF$prediction)

out_test_year <- test_df %>%
  select(county_fips, any_of(OUTCOME)) %>%  mutate(pred_outcome = pred_testyr) 

eval_test_year <- out_test_year %>% drop_na(all_of(OUTCOME))

r_test    <- r_fun(eval_test_year$pred_outcome, eval_test_year[[OUTCOME]])
r2_test   <- r2(eval_test_year$pred_outcome, eval_test_year[[OUTCOME]])
mae_test  <- mae(eval_test_year$pred_outcome, eval_test_year[[OUTCOME]])
rmse_test <- rmse(eval_test_year$pred_outcome, eval_test_year[[OUTCOME]])

# cat("\n--- GWR TRANSFER (training → transfer year, gwr.predict) ---\n")
# cat("r   :", round(r_test, 3), "\n")
# cat("R2  :", round(r2_test, 3), "\n")
# cat("MAE :", round(mae_test, 3), "\n")
# cat("RMSE:", round(rmse_test, 3), "\n")

#save transfer year predictions
#write.csv(out_test_year, paste0("./Scripts/GWR/gwrorig_t2_", OUTCOME, ".csv"), row.names = FALSE)

# ============================================================
# ADD-ON: Update Transfer Year Predictions Using Predicted Obesity and Smoking Values
# ============================================================
#read predicted obesity and smoking from previous model outputs
pred_obesity <- read.csv("./Scripts/GWR/gwrorig_t2_per_obesit.csv") %>%
  select(county_fips, pred_outcome) %>%
  mutate(
    county_fips = pad5(county_fips),
    pred_outcome = to_num(pred_outcome)
  ) %>%
  rename(per_obesit = pred_outcome)

pred_smoking <- read.csv("./Scripts/GWR/gwrorig_t2_per_smokin.csv") %>%
  select(county_fips, pred_outcome) %>%
  mutate(
    county_fips = pad5(county_fips),
    pred_outcome = to_num(pred_outcome)
  ) %>%
  rename(per_smokin = pred_outcome)

#replace original transfer year obesity and smoking with predicted values
test_df_updated <- test_year$df %>%
  select(-per_obesit, -per_smokin) %>%
  left_join(pred_obesity, by = "county_fips") %>%
  left_join(pred_smoking, by = "county_fips") %>%
  drop_na(all_of(COMMON_VARS))

coords_test_updated <- test_year$coords[test_df_updated$.rid, , drop = FALSE]

sp_test_updated <- SpatialPointsDataFrame(
  coords      = coords_test_updated,
  data        = test_df_updated[, c("county_fips", OUTCOME, COMMON_VARS), drop = FALSE],
  proj4string = crs_obj
)

gwr_pred_updated <- GWmodel::gwr.predict(
  formula     = form,
  data        = sp_train,
  predictdata = sp_test_updated,
  bw          = k_use,
  kernel      = "bisquare",
  adaptive    = TRUE,
  longlat     = FALSE
)

SDF_updated <- as.data.frame(gwr_pred_updated$SDF)

pred_testyr_updated <- as.numeric(SDF_updated$prediction)

out_test_year_updated <- test_df_updated %>%
  select(county_fips, any_of(OUTCOME)) %>%
  mutate(pred_outcome = pred_testyr_updated)

#save final predictions
# write.csv(
#   out_test_year_updated,
#   paste0("./Scripts/GWR/gwrorig_t2_", OUTCOME, "_updated.csv"),
#   row.names = FALSE
# )

#updated evaluation
eval_test_year_updated <- out_test_year_updated %>% drop_na(all_of(OUTCOME))

cat("\n--- GWR original: Train Year → Transfer Year with Predicted Obesity + Smoking ---\n")
cat("r   :", round(r_fun(eval_test_year_updated$pred_outcome,
                         eval_test_year_updated[[OUTCOME]]), 3), "\n")
cat("R2  :", round(r2(eval_test_year_updated$pred_outcome,
                      eval_test_year_updated[[OUTCOME]]), 3), "\n")
cat("MAE :", round(mae(eval_test_year_updated$pred_outcome,
                       eval_test_year_updated[[OUTCOME]]), 3), "\n")
cat("RMSE:", round(rmse(eval_test_year_updated$pred_outcome,
                        eval_test_year_updated[[OUTCOME]]), 3), "\n")