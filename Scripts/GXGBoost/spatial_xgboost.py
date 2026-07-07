# ============================================================
# G-XGBoost
# ============================================================
# NOTE:
# First, exclude obesity and smoking from VARS and run the script until the
# add-on section to generate obesity and smoking predictions.
#
# Then, include obesity and smoking in VARS and run the full script to
# generate the final chronic disease predictions.
# ============================================================

import numpy as np
import pandas as pd
import geopandas as gpd

from geoxgboost.geoxgboost import gxgb
from scipy.spatial import distance_matrix

from sklearn.metrics import mean_squared_error, r2_score, mean_absolute_error


OUTCOME = "per_copd"    #change based on what you're predicting

train_year_path = "./Data/2019ACS_2023cdcPlaces_2019CHR.shp"
test_year_path = "./Data/2022ACS_2024cdcPlaces_2022CHR.shp"

VARS = [
    "over_65",
    "white","female","bach","uninsured","poverty",
    "per_obesit","per_smokin","per_rural"
]

CRS_EPSG = 5070
RANDOM_SEED = 123


def pad5(series: pd.Series) -> pd.Series:
    """Zero-pad county FIPS / GEOID to 5 chars."""
    return series.astype(str).str.zfill(5)


def to_num(series: pd.Series) -> pd.Series:
    return pd.to_numeric(series, errors="coerce")


def rmse(y_true, y_pred) -> float:
    return float(np.sqrt(mean_squared_error(y_true, y_pred)))


def mae(y_true, y_pred) -> float:
    return float(mean_absolute_error(y_true, y_pred))


def r_corr(y_true, y_pred) -> float:
    return float(pd.Series(y_true).corr(pd.Series(y_pred)))


def load_year_with_coords(
    shp_path: str,
    fips_col: str,
    outcome_col: str
) -> pd.DataFrame:

    df = gpd.read_file(shp_path)

    df = df.rename(columns={fips_col: "county_fips"})
    df["county_fips"] = pad5(df["county_fips"]) 

    df = df.to_crs(CRS_EPSG)
    df["X"] = df.geometry.centroid.x
    df["Y"] = df.geometry.centroid.y

    for v in VARS:
        if v in df.columns: 
            df[v] = to_num(df[v])
            df.loc[df[v] == -999, v] = np.nan

    if outcome_col in df.columns:
        df[outcome_col] = to_num(df[outcome_col])
        df.loc[df[outcome_col] == -999, outcome_col] = np.nan

    cols = ["county_fips", "X", "Y"] + [v for v in VARS if v in df.columns]

    if outcome_col in df.columns:
        cols.append(outcome_col)

    df = df[cols].copy()

    return df


def gxgb_predict_no_save(
    X_pred: pd.DataFrame,
    coords_pred: pd.DataFrame,
    train_coords: pd.DataFrame,
    gxgb_result: dict,
    alpha_wt: float = 0.5,
    alpha_wt_type: str = "fixed"
) -> np.ndarray:


    # Distance matrix (n_pred x n_train)
    dist = distance_matrix(coords_pred.values, train_coords.values)
    index_min = dist.argmin(axis=1) # nearest training unit for each pred unit

    Alpha_wtDF = pd.DataFrame(gxgb_result["alpha_wt"])
    y_G_hat = pd.DataFrame(gxgb_result["y_G_hat"])
    best_local_models = gxgb_result["bestLocalModel"]

    preds = []

    if alpha_wt_type not in ["fixed", "varying"]:
        raise ValueError("alpha_wt_type must be 'fixed' or 'varying'.")

    for i in range(X_pred.shape[0]):
        idx = int(index_min[i])

        local_model = best_local_models[idx]
        x_i = pd.DataFrame(X_pred.iloc[i, :]).T

        # local prediction
        y_loc = float(local_model.predict(x_i)[0])

        # global prediction at that training location
        y_glob = float(y_G_hat.iloc[idx, 0])

        if alpha_wt_type == "fixed":
            a = alpha_wt
        else:
            a = float(Alpha_wtDF.iloc[idx, 0])

        b = 1.0 - a
        preds.append(a * y_loc + b * y_glob)

    return np.array(preds)


#main
if __name__ == "__main__":
    np.random.seed(RANDOM_SEED)


    print("=== Loading training year data for G-XGBoost ===")
    df_train = load_year_with_coords(
        shp_path=train_year_path,
        fips_col="CountyFIPS",
        outcome_col=OUTCOME
    )

    print("=== Loading transfer year data for temporal eval ===")
    df_test = load_year_with_coords(
        shp_path=test_year_path,
        fips_col="CountyFIPS",
        outcome_col=OUTCOME
    )

    common_fips = set(df_train["county_fips"]).intersection(set(df_test["county_fips"]))

    df_train = df_train[df_train["county_fips"].isin(common_fips)].copy()
    df_test = df_test[df_test["county_fips"].isin(common_fips)].copy()


    m = (
        df_train[["county_fips", OUTCOME]]
        .rename(columns={OUTCOME: f"{OUTCOME}_train"})
        .merge(
            df_test[["county_fips", OUTCOME]].rename(columns={OUTCOME: f"{OUTCOME}_test"}),
            on="county_fips",
            how="inner"
        )
        .dropna()
    )

    if len(m) > 0:
        corr = m[f"{OUTCOME}_train"].corr(m[f"{OUTCOME}_test"])
        print(f"\n=== Cross-year correlation of {OUTCOME} ===")
        print(f"Pearson r: {corr:.3f}")
    else:
        print(f"\n[WARN] No overlapping non-missing {OUTCOME} between train and transfer year.")


    needed_train = ["X", "Y"] + VARS + [OUTCOME]
    needed_train = [c for c in needed_train if c in df_train.columns]

    df_train_mod = df_train.dropna(subset=needed_train).reset_index(drop=True)
    print(f"\n[Training Year] usable rows (full coords + predictors + outcome): {len(df_train_mod)}")

    X_train = df_train_mod[VARS].astype(float)
    y_train = df_train_mod[[OUTCOME]].astype(float) # DataFrame (n x 1)
    Coords_train = df_train_mod[["X", "Y"]].astype(float)


    params = {
        "n_estimators": 1500,
        "max_depth": 3,
        "learning_rate": 0.05,
        "subsample": 0.8,
        "colsample_bytree": 0.8,
        "min_child_weight": 5,
        "reg_lambda": 10,
        "reg_alpha": 0.1,   # MUST exist or gxgb throws KeyError
        "gamma": 0.0,
        "random_state": RANDOM_SEED
    }

    bw_neighbors = 200

    print(f"\n=== Running Geographical XGBoost on Training Year ({OUTCOME}) ===")
    gxgb_out = gxgb(
        X=X_train,
        y=y_train,
        Coords=Coords_train,
        params=params,
        bw=bw_neighbors,
        Kernel="Adaptive",
        spatial_weights=True,
        alpha_wt=0.5,            # ensemble between local and global
        alpha_wt_type="fixed", 
        feat_importance="gain",
        test_size=0.2,
        seed=RANDOM_SEED,
        path_save=False          
    )

    needed_test = ["X", "Y"] + VARS
    needed_test = [c for c in needed_test if c in df_test.columns]

    df_test_pred = df_test.dropna(subset=needed_test).reset_index(drop=True)
    print(f"\n[Transfer Year] rows with full coords + predictors: {len(df_test_pred)}")

    X_test = df_test_pred[VARS].astype(float)
    Coords_test = df_test_pred[["X", "Y"]].astype(float)

    print(f"\n=== Predicting {OUTCOME} Transfer Year with trained G-XGBoost ===")
    y_test_pred = gxgb_predict_no_save(
        X_pred=X_test,
        coords_pred=Coords_test,
        train_coords=Coords_train,
        gxgb_result=gxgb_out,
        alpha_wt=0.5,
        alpha_wt_type="fixed"
    )

    df_test_pred["pred_outcome"] = y_test_pred

    out_test_year = df_test_pred[["county_fips", OUTCOME, "pred_outcome"]].copy()

    # Evaluate only where true outcome is available
    eval_test_year = out_test_year.dropna(subset=[OUTCOME]).copy()

    if len(eval_test_year) > 0:
        y_true = eval_test_year[OUTCOME].astype(float).values
        y_hat = eval_test_year["pred_outcome"].astype(float).values

        r_val = r_corr(y_true, y_hat)
        r2_val = float(r2_score(y_true, y_hat))
        mae_val = mae(y_true, y_hat)
        rmse_val = rmse(y_true, y_hat)

        #print(f"\n--- TRANSFER YEAR TEMPORAL VALIDATION (G-XGBoost, {OUTCOME}) ---")
        #print(f"r   : {r_val:.3f}")
        #print(f"R2  : {r2_val:.3f}")
        #print(f"MAE : {mae_val:.3f}")
        #print(f"RMSE: {rmse_val:.3f}")

    else:
        print(f"\n[Transfer Year] No non-missing {OUTCOME} values for temporal evaluation.")

    # save transfer year predictions
    #out_test_year.to_csv(f"./Scripts/GXGBoost/gxgboost_t2_{OUTCOME}.csv", index=False)

    # ============================================================
    # ADD-ON: Update Transfer Year Predictions Using Predicted Obesity and Smoking Values
    # ============================================================

    pred_obesity = (
        pd.read_csv("./Scripts/GXGBoost/gxgboost_t2_per_obesit.csv")
        [["county_fips", "pred_outcome"]]
        .copy()
    )
    pred_obesity["county_fips"] = pad5(pred_obesity["county_fips"])
    pred_obesity["pred_outcome"] = to_num(pred_obesity["pred_outcome"])
    pred_obesity = pred_obesity.rename(columns={"pred_outcome": "per_obesit"})

    pred_smoking = (
        pd.read_csv("./Scripts/GXGBoost/gxgboost_t2_per_smokin.csv")
        [["county_fips", "pred_outcome"]]
        .copy()
    )
    pred_smoking["county_fips"] = pad5(pred_smoking["county_fips"])
    pred_smoking["pred_outcome"] = to_num(pred_smoking["pred_outcome"])
    pred_smoking = pred_smoking.rename(columns={"pred_outcome": "per_smokin"})


    df_test_updated = (
        df_test
        .drop(columns=["per_obesit", "per_smokin"])
        .merge(pred_obesity, on="county_fips", how="left")
        .merge(pred_smoking, on="county_fips", how="left")
        .dropna(subset=needed_test)
    )

    X_test_updated = df_test_updated[VARS].astype(float)
    Coords_test_updated = df_test_updated[["X", "Y"]].astype(float)

    print(f"\n=== Predicting {OUTCOME} Transfer Year with Predicted Obesity + Smoking ===")
    y_test_updated_pred = gxgb_predict_no_save(
        X_pred=X_test_updated,
        coords_pred=Coords_test_updated,
        train_coords=Coords_train,
        gxgb_result=gxgb_out,
        alpha_wt=0.5,
        alpha_wt_type="fixed"
    )

    df_test_updated["pred_outcome"] = y_test_updated_pred

    out_test_year_updated = df_test_updated[["county_fips", OUTCOME, "pred_outcome"]].copy()
    
    # Evaluate only where true outcome is available
    eval_test_year_updated = out_test_year_updated.dropna(subset=[OUTCOME]).copy()

    if len(eval_test_year_updated) > 0:
        y_true_updated = eval_test_year_updated[OUTCOME].astype(float).values
        y_hat_updated = eval_test_year_updated["pred_outcome"].astype(float).values

        r_updated = r_corr(y_true_updated, y_hat_updated)
        r2_updated = float(r2_score(y_true_updated, y_hat_updated))
        mae_updated = mae(y_true_updated, y_hat_updated)
        rmse_updated = rmse(y_true_updated, y_hat_updated)

        print(f"\n--- TRANSFER YEAR TEMPORAL VALIDATION WITH PREDICTED OBESITY + SMOKING (G-XGBoost, {OUTCOME}) ---")
        print(f"r   : {r_updated:.3f}")
        print(f"R2  : {r2_updated:.3f}")
        print(f"MAE : {mae_updated:.3f}")
        print(f"RMSE: {rmse_updated:.3f}")

    else:
        print(f"\n[Transfer Year Updated] No non-missing {OUTCOME} values for temporal evaluation.")

    # save final predictions
    #out_test_year_updated.to_csv(f"./Scripts/GXGBoost/gxgboost_t2_{OUTCOME}_updated.csv", index=False)
