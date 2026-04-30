# =============================================================================
# background code.R
# =============================================================================

# This script trains and saves prediction models for every possible combination
# of the 5 available predictor groups. The resulting .rds files are what app.R 
# loads at runtime to make predictions.

# This is NOT part of the actual Shiny app! It only needs to be run once, any time
# the models need to be rebuilt (like when adding a new model type)
# The output files are placed in the app directory and deployed to the web along 
# with app.R. 

# HOW TO RUN
# 1. set data_dir (Section 1) to your folder containing 09_Data_cleaned_final.Rds
# 2. set model_out_dir (Section 6) to your local copy of the app directory
# 3. run the entire script. It takes a few minutes.
# 4. verify the output directory contains .rds files before deploying app.R to web

# OUTPUT FILES
# for each of the 31 predictor combinations, this script saves:
#   mod_<combo_tag>_lm.rds     — fitted lm object
#   mod_<combo_tag>_rf.rds     — fitted randomForest object
#   meta_<combo_tag>.rds       — list with RMSE scores and best model name

# ADDING NEW MODELS
# 1. fit your new model inside the loop in Section 7, following the lm/rf
#    pattern
# 2. the meta file will automatically pick it up as a candidate for best_model

# =============================================================================

library(tidyverse)
library(randomForest)

# =============================================================================
# SECTION 1: load cleaned dataset
# =============================================================================
# 09_Data_cleaned_final.Rds is the output of Changfei's data cleaning pipeline.
# if you use a different dataset, confirm the variable names & structure match 
# what is expected below

# *** UPDATE THIS PATH to where your dataset is stored 
data_clean <- readRDS("C:/Users/natal/Downloads/data_process/data_process/09_Data_cleaned_final.Rds")

# select & rename only the variables needed for modeling
# Changfei's cleaned data contains additional variables that were considered 
# during his stepwise model selection but ultimately excluded from his final 
# predictive model. To keep the modeling clean, I used only the 5 predictor groups 
# and the response
data <- data_clean %>%
  transmute(
    Compound_time     = Comp_time_batch,         # response: compounding time in minutes
    Vials             = Sum_count_product,       # total number of product vials per dose
    Transfer_volume   = Sum_Transfer_volume,     # total mL transferred per dose
    Difficulty        = Difficulty,              # 1 (easy) to 5 (hard) see Table 1 in Changfei's report
    Batch             = Batch,                   # 0 = single dose, 1 = batch dose
    Beds              = Beds,                    # facility type
  )

# correct typo: "Output" should be "Outpatient"
# needs to be fixed before factorizing Beds or "Outpatient" level will get missed
data$Beds <- case_when(
  data$Beds == "Output" ~ "Outpatient",
  TRUE ~ data$Beds
)


# =============================================================================
# SECTION 2: factorize, log-transform, dummy-code Beds
# =============================================================================
# all transformations applied here must be mirrored in app.R's predict_time()
# function. if you change anything in this section, update predict_time() too.

# factorize categorical variables before log-transforming
data$Batch <- factor(data$Batch)
data$Beds  <- factor(data$Beds)

# log-transform continuous variables to reduce right skew
data_tr <- data %>%
  mutate(
    Compound_time     = log(Compound_time),
    Vials             = log(Vials),
    Transfer_volume   = log(Transfer_volume + 0.1)
    # +0.1 prevents log(0) for doses with zero transfer volume.
    # this offset must also be applied in app.R when building newdata
  )

# dummy-code Beds rather than passing it as a factor to the models
# each column is named Beds_<level> (eg Beds_Cancer, Beds_100)
beds_levels <- levels(data_tr$Beds)
for (b in beds_levels) {
  colname <- paste0("Beds_", b)
  data_tr[[colname]] <- as.numeric(data_tr$Beds == b)
}
data_tr <- data_tr %>% select(-Beds)  # drop the original Beds factor column


# =============================================================================
# SECTION 3: train/test split
# =============================================================================
# 90% of observations go to training, 10% to the held-out test set
#   (with 114k observations, even 10% gives 11,400 test cases which is enough 
#   for reasonably stable RSME estimates)
# set.seed(3) ensures reproducibility. DO NOT change this seed without also
# retraining all models!

set.seed(3)
n         <- nrow(data_tr)
train_idx <- sample(seq_len(n), size = floor(0.9 * n))

train_data <- data_tr[train_idx, ]
test_data  <- data_tr[-train_idx, ]


# =============================================================================
# SECTION 4: define predictor groups
# =============================================================================

# the NAMES of this list define the labels used in combo_tag strings
# the ORDER of the names determines the canonical order of those labels in combo_tags
# if you add a new predictor group or rename an existing one, you must also update:
#   - the provided <- c() block in app.R's predict_time()
#   - the map_facility_to_beds() function if the new group is facility-related
#   - the newrow construction block in app.R's predict_time()
#   - the UI inputs if the new predictor needs a user-facing control

predictor_groups <- list(
  Vials           = "Vials",
  Transfer_volume = "Transfer_volume",
  Difficulty      = "Difficulty",
  Batch           = "Batch",
  Beds            = grep("^Beds_", names(train_data), value = TRUE)
  # grep() dynamically finds all Beds_* columns so this stays correct
  # even if the number of Beds levels changes in a future dataset
)

group_names <- names(predictor_groups)

# generate all 31 non-empty subsets of the 5 predictor groups
# combn(group_names, k) gives all subsets for k = 1 through 5
# and unlist into a single flat list of character vectors
all_combos <- unlist(
  lapply(1:length(group_names), function(k) combn(group_names, k, simplify = FALSE)),
  recursive = FALSE
)


# =============================================================================
# SECTION 5: compute metrics
# =============================================================================
# computed on the back-transformed scale in minutes for interpretability

compute_metrics <- function(true_log, pred_log) {
  true <- exp(true_log)   # back-transform from log scale
  pred <- exp(pred_log)
  rmse <- sqrt(mean((true - pred)^2))
  mae  <- mean(abs(true - pred))
  mape <- mean(abs((true - pred) / true))
  list(RMSE = rmse, MAE = mae, MAPE = mape)  
  # MAPE is a proportion, not a percentage: 0.55 = 55% mean absolute error
}


# =============================================================================
# SECTION 6: set output directory
# =============================================================================
# model_out_dir must be the root of the Shiny app directory. all model and
# metadata files saved here will be read by app.R at runtime 

# make sure the app.R file and these model/metadata files are the ONLY things 
# in this folder, or they won't deploy to the web properly. 
# background code.R and source datasets must be stored elsewhere!

# *** UPDATE THIS PATH before running on a new machine
model_out_dir <- "C:/Users/natal/OneDrive/Desktop/STA698_app/"
if (!dir.exists(model_out_dir)) dir.create(model_out_dir, recursive = TRUE)


# =============================================================================
# SECTION 7: loop through model training
# =============================================================================
# for each of the 31 predictor combinations, this loop:
#   1. builds a formula using the expanded column names for that combo
#   2. fits a linear model (lm) and a random forest (rf) on training data
#   3. evaluates both on test data using compute_metrics()
#   4. selects the model with the lower test RMSE as "best"
#   5. saves both model objects and a metadata list to the output folder

# REMEMBER to re-run any time you make changes to update the models saved in the directory!

for (combo in all_combos) {
  
  # expand group names to actual column names
  # ex: combo = c("Vials","Beds") -> vars = c("Vials","Beds_Cancer","Beds_100",...)
  vars      <- unlist(predictor_groups[combo])
  combo_tag <- paste(combo, collapse = "_")   #ex: "Vials_Beds"
  
  # build the model formula as a string and convert to a formula object
  fmla <- as.formula(paste("Compound_time ~", paste(vars, collapse = " + ")))
  
  
  # MODEL FITTING ------------------------------------------------------------
  # --- linear model ---
  lm_fit <- tryCatch(
    lm(fmla, data = train_data),
    error = function(e) {
      message("lm failed for combo: ", combo_tag, " — ", conditionMessage(e))
      NULL
    }
  )
  
  
  # --- random forest ---
  # ntree = 200: a modest forest size chosen for speed over a full 500-tree forest 
  # increasing ntree may improve accuracy but will slow down both training & web deployment 
  # 200 seemed like enough for placeholder models
  rf_fit <- tryCatch(
    randomForest(fmla, data = train_data, ntree = 200),
    error = function(e) {
      message("rf failed for combo: ", combo_tag, " — ", conditionMessage(e))
      NULL
    }
  )
  
  
  # --- ADD NEW MODEL HERE ---
  # newmodel_fit <- tryCatch(
    # newmodel(),
    # error = function(e) {
      # message("newmodel failed for combo: ", combo_tag, " — ", conditionMessage(e))
      # NULL
    #}
  #)
  
  
  
  # --- evaluate on test data ---
  lm_metrics <- NULL
  rf_metrics <- NULL
  # newmodel_metrics <- NULL
  
  if (!is.null(lm_fit)) {
    lm_pred <- tryCatch(predict(lm_fit, newdata = test_data), error = function(e) NA)
    if (!any(is.na(lm_pred)))
      lm_metrics <- compute_metrics(test_data$Compound_time, lm_pred)
  }
  
  if (!is.null(rf_fit)) {
    rf_pred <- tryCatch(predict(rf_fit, newdata = test_data), error = function(e) NA)
    if (!any(is.na(rf_pred)))
      rf_metrics <- compute_metrics(test_data$Compound_time, rf_pred)
  }
  
  #if (!is.null(newmodel_fit)) {
  #  newmodel_pred <- tryCatch(predict(newmodel_fit, newdata = test_data), error = function(e) NA)
  #  if (!any(is.na(newmodel_pred)))
  #    newmodel_metrics <- compute_metrics(test_data$Compound_time, newmodel_pred)
  #}
  
  
  
  # --- select best model by test RMSE ---
  # rmse_by_model is a list so that future model types can be added by adding another entry
  # which.min() selects the name with the lowest RMSE
  # NA values are automatically excluded by unlist()
  rmse_by_model <- list(
    lm = if (!is.null(lm_metrics)) lm_metrics$RMSE else NA,
    rf = if (!is.null(rf_metrics)) rf_metrics$RMSE else NA #,
    # newmodel = if (!is.null(newmodel_metrics)) newmodel_metrics$RMSE else NA
  )
  
  best_model <- names(which.min(unlist(rmse_by_model)))
  best_rmse  <- min(unlist(rmse_by_model), na.rm = TRUE)
  
  # --- save model objects ---
  # file naming convention: mod_<combo_tag>_<model_type>.rds
  # app.R and get_all_model_preds() both depend on this EXACT naming pattern
  if (!is.null(lm_fit))
    saveRDS(lm_fit, file.path(model_out_dir, paste0("mod_", combo_tag, "_lm.rds")))
  if (!is.null(rf_fit))
    saveRDS(rf_fit, file.path(model_out_dir, paste0("mod_", combo_tag, "_rf.rds")))
  
  # --- save metadata ---
  # the metadata file is what app.R reads first to determine which model to use
  # it stores all metric results for all model types
  meta <- list(
    combo         = combo,           # character vector of group names
    combo_tag     = combo_tag,       # underscore-joined string, used in filenames
    lm_metrics    = lm_metrics,      # list(RMSE, MAE, MAPE) for lm, or NULL
    rf_metrics    = rf_metrics,      # list(RMSE, MAE, MAPE) for rf, or NULL
    rmse_by_model = rmse_by_model,   # named list used by app.R's model selection
    best_model    = best_model,      # whoever had lower test RMSE
    best_rmse     = best_rmse,       # that model's test RMSE in minutes
    created_at    = Sys.time()       # timestamp for provenance tracking
  )
  
  saveRDS(meta, file.path(model_out_dir, paste0("meta_", combo_tag, ".rds")))
}

cat("Model training complete. Files saved to:", model_out_dir, "\n")
cat("Total combos trained:", length(all_combos), "\n") 
# always double check that you have the expected number of output files!
# in this case: (2 * 31 mod) + 31 meta = 93 files