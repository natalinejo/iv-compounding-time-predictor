# =============================================================================
# app.R
# =============================================================================

# This app lets a user enter any combination of up to 5 compounding-related
# inputs and receive a predicted compounding time in minutes. It runs the models
# trained and saved by background code.R.

# HOW IT WORKS
# 1. the user fills in whichever inputs they have. Blank = unknown
# 2. the app detects which inputs were provided and constructs a "combo_tag"
#    string naming that exact combination (ex: "Vials_Batch" or
#    "Vials_Transfer_volume_Difficulty_Batch_Beds")
# 3. it looks for pre-trained model files matching that combo_tag in the directory.
#    After background code.R is run, it will have already trained one lm and one
#    rf model for every possible non-empty combination of the five predictor groups,
#    so there is always a model available for whatever the user provides
# 4. it picks the model with the lower test RMSE (stored in a companion
#    metadata file in the directory), runs the prediction, back-transforms from
#    log scale, and displays the result

# APP DIRECTORY LAYOUT
#   app.R                              <- this file
#   mod_<combo_tag>_lm.rds             <- saved lm model objects (one per combo)
#   mod_<combo_tag>_rf.rds             <- saved rf model objects (one per combo)
#   meta_<combo_tag>.rds               <- metadata: RMSE scores, best model name

# WEB DEPLOYMENT
# run from the R console (NOT from inside the app itself):
#   rsconnect::deployApp("C:/Users/natal/OneDrive/Desktop/STA698_app/")
# see README.md for step-by-step first-deployment instructions

# =============================================================================

library(shiny)          # the web application framework
library(rsconnect)      # used for deployment. not needed to run app locally
library(randomForest)
library(ggplot2)
library(dplyr)


# =============================================================================
# HELPER FUNCTION: map_facility_to_beds()
# =============================================================================

# the training data encodes hospital size as a "Beds" category (a string like
# "100", "500", "Cancer", or "Outpatient")
# the UI presents friendlier labels like "Small Hospital" or "Cancer Center"
# this function translates between the two so the rest of the app can work in
# training-data terms

# the final bare NA (the default case of switch()) handles the blank " "
# selection in the UI. returning NA signals that facility type was not provided,
# so "Beds" will be excluded from the combo_tag

# TO ADD A NEW FACILITY TYPE
# add a line here AND add the matching label to the selectInput() choices list
# in the UI section below

map_facility_to_beds <- function(facility_type) {
  switch(
    facility_type,
    "Cancer Center"                = "Cancer",
    "Large Academic Medical Center" = "1000",
    "Large Hospital"               = "500",
    "Medium Hospital"              = "300",
    "Small Hospital"               = "200",
    "Community Hospital"           = "100",
    "Women's Hospital"             = "125",
    "Outpatient"                   = "Outpatient",
    "N/A"                          = NA,
    NA   # default: blank " " or unrecognized value -> treat as not provided
  )
}


# =============================================================================
# HELPER FUNCTION: get_all_model_preds()
# =============================================================================

# used ONLY in Developer Mode for a given combo_tag and a prepared newdata row
# this function finds every saved model file matching that combo (both lm
# and rf, plus any future model types added later), runs each one, & returns
# a data frame summarizing each model's prediction & test RMSE. This lets you
# compare all available models side-by-side for any input combination

# if a model file fails to load or requires predictors not present in newdata,
# it records that in the "status" column instead of just crashing

get_all_model_preds <- function(combo_tag, newdata) {
  # find all model files matching this combo_tag, regardless of model type.
  # file naming convention: mod_<combo_tag>_<model_type>.rds
  files <- list.files(
    path       = "./",
    pattern    = paste0("^mod_", combo_tag, "_.*\\.rds$"),
    full.names = TRUE
  )
  if (length(files) == 0)
    return(NULL)
  
  out <- lapply(files, function(f) {
    # extract the model type label (e.g. "lm", "rf") from the filename
    model_name <- sub(paste0("^mod_", combo_tag, "_(.*)\\.rds$"),
                      "\\1",
                      basename(f))
    
    # attempt to load model --> return a failure row if the file can't be read
    model_obj <- tryCatch(
      readRDS(f),
      error = function(e)
        NULL
    )
    if (is.null(model_obj)) {
      return(data.frame(
        model = model_name,
        status = "failed to load",
        pred = NA,
        rmse = NA
      ))
    }
    
    # determine which predictor columns this model expects
    # lm objects expose their formula, randomForest exposes $xNames
    fmla <- tryCatch(
      formula(model_obj),
      error = function(e)
        NULL
    )
    if (!is.null(fmla)) {
      req <- all.vars(fmla)[-1]           # [-1] drops the response variable
    } else if (!is.null(model_obj$xNames)) {
      req <- model_obj$xNames
    } else {
      req <- character(0)
    }
    
    # if newdata is missing any required column, mark as not applicable
    missing <- setdiff(req, names(newdata))
    if (length(missing) > 0) {
      return(data.frame(
        model  = model_name,
        status = paste0(
          "not applicable (missing: ",
          paste(missing, collapse = ", "),
          ")"
        ),
        pred   = NA,
        rmse   = NA
      ))
    }
    
    # predict on log scale, then back-transform to minutes.
    # randomForest's predict method must be called explicitly via getS3method()
    # because plain predict() can dispatch to the wrong method in some
    # environments when multiple modeling packages are loaded simultaneously.
    pred_log <- tryCatch({
      if (inherits(model_obj, "randomForest"))
        getS3method("predict", "randomForest")(model_obj, newdata = newdata)
      else
        predict(model_obj, newdata = newdata)
    }, error = function(e)
      NA)
    
    pred <- ifelse(is.na(pred_log), NA, exp(pred_log))
    
    # pull this model's test RMSE from the shared metadata file for this combo
    meta_file <- file.path(dirname(f), paste0("meta_", combo_tag, ".rds"))
    rmse <- NA
    if (file.exists(meta_file)) {
      meta <- tryCatch(
        readRDS(meta_file),
        error = function(e)
          NULL
      )
      if (!is.null(meta$rmse_by_model) &&
          model_name %in% names(meta$rmse_by_model))
        rmse <- meta$rmse_by_model[[model_name]]
    }
    
    data.frame(
      model = model_name,
      status = "applicable",
      pred = pred,
      rmse = rmse
    )
  })
  
  do.call(rbind, out)
}


# =============================================================================
# CORE FUNCTION: predict_time()
# =============================================================================

# GIVEN a named list of user inputs, predict_time():
#   1. determines which inputs were actually provided (non-NA)
#   2. builds a combo_tag string identifying that exact predictor combination
#   3. applies the same transformations used during training so the prediction
#      input matches what the model expects
#   4. loads the best-performing pre-trained model for this combo from the directory
#   5. returns the prediction (back-transformed from log scale) plus metadata

# RETURNS a named list with elements:
#   $pred           <-- predicted compounding time in minutes (numeric), or NA
#   $model          <-- human-readable name of the model used (character)
#   $test_rmse      <-- test-set RMSE of the chosen model in minutes (numeric)
#   $combo_tag      <-- the predictor combination string used (character)
#   $debug_newdata  <-- the prepared single-row data frame sent to predict()

predict_time <- function(inputs) {
  # models are saved in the app's working directory
  # "./" resolves correctly both when running locally & on shinyapps.io via rsconnect
  # (shinyapps.io gets angry if you try to hardcode this path to your machine)
  model_out_dir <- "./"
  
  # unpack raw inputs
  vials_val      <- inputs$vials
  transfer_val   <- inputs$transfer
  difficulty_val <- if (!is.na(inputs$difficulty))
    inputs$difficulty
  else
    NA_real_
  batch_val      <- if (!is.na(inputs$batch))
    inputs$batch
  else
    NA_real_
  beds_cat       <- map_facility_to_beds(inputs$facility_type_raw)
  
  # -------------------------------------------------------------------
  # STEP 1: determine which predictor groups are available
  # -------------------------------------------------------------------
  # "provided" holds the group names for any input the user actually supplied.
  # THE ORDER HERE MUST MATCH the order of predictor_groups in background_code.R,
  # because combo_tag is built by pasting these names together
  # if the order is not the same, the generated tag won't match the saved file names
  # current canonical order:
  #           Vials -> Transfer_volume -> Difficulty -> Batch -> Beds
  
  provided <- c()
  if (!is.na(vials_val))
    provided <- c(provided, "Vials")
  if (!is.na(transfer_val))
    provided <- c(provided, "Transfer_volume")
  if (!is.na(difficulty_val))
    provided <- c(provided, "Difficulty")
  if (!is.na(batch_val))
    provided <- c(provided, "Batch")
  if (!is.na(beds_cat))
    provided <- c(provided, "Beds")
  
  if (length(provided) < 1) {
    return(
      list(
        pred          = NA,
        model         = "No predictors provided",
        test_rmse     = NA,
        combo_tag     = NA,
        debug_newdata = NULL
      )
    )
  }
  
  # combo_tag example: "Vials_Transfer_volume_Batch"
  # this string is used to find model files: mod_Vials_Transfer_volume_Batch_rf.rds
  combo_tag <- paste(provided, collapse = "_")
  meta_file <- file.path(model_out_dir, paste0("meta_", combo_tag, ".rds"))
  
  # -------------------------------------------------------------------
  # STEP 2: build the single-row prediction data frame (newdata)
  # -------------------------------------------------------------------
  # all transformations here mirror those applied in background_code.R Section 2.
  # if you change the training-time transformations, you must update this block
  # to match or predictions will be on the wrong scale
  
  newrow <- list()
  
  if ("Vials" %in% provided)
    # log transform matches training. Vials is always >= 1, so log(Vials) >= 0
    newrow$Vials <- log(vials_val)
  
  if ("Transfer_volume" %in% provided)
    # +0.1 offset before log prevents log(0) for zero-volume transfers
    # this same offset was applied at training time so must stay here
    newrow$Transfer_volume <- log(transfer_val + 0.1)
  
  if ("Difficulty" %in% provided)
    # Difficulty (1-5) is used as a numeric predictor, not a factor
    newrow$Difficulty <- difficulty_val
  
  if ("Batch" %in% provided)
    # Batch is 0/1 numeric here -- it's coerced to a factor with levels c(0, 1)
    # 2 lines below, matches how it was stored in the training data
    newrow$Batch <- batch_val
  
  if ("Beds" %in% provided) {
    # Beds is a nominal categorical variable w/ 8 levels.
    # instead of passing it as a factor (risking level-mismatch errors), it's
    # manually one-hot encoded into 8 binary columns.
    
    # all 8 dummies are included even though only one will be 1 for any given prediction
    
    # lm and randomForest both silently ignore columns they don't use, so extra
    # columns are harmless.
    # including all 8 dummies means any future model type can reference any
    # Beds level without requiring changes here.
    newrow$Beds_Cancer     <- as.numeric(beds_cat == "Cancer")
    newrow$Beds_100        <- as.numeric(beds_cat == "100")
    newrow$Beds_125        <- as.numeric(beds_cat == "125")
    newrow$Beds_200        <- as.numeric(beds_cat == "200")
    newrow$Beds_300        <- as.numeric(beds_cat == "300")
    newrow$Beds_500        <- as.numeric(beds_cat == "500")
    newrow$Beds_1000       <- as.numeric(beds_cat == "1000")
    newrow$Beds_Outpatient <- as.numeric(beds_cat == "Outpatient")
  }
  
  newdata <- as.data.frame(newrow, stringsAsFactors = FALSE)
  
  # coerce Batch to a factor with the exact same levels used during training.
  # this prevents predict() from throwing "factor levels do not match" warnings
  if ("Batch" %in% names(newdata))
    newdata$Batch <- factor(newdata$Batch, levels = c(0, 1))
  
  # -------------------------------------------------------------------
  # STEP 3: locate the best model for this combo
  # -------------------------------------------------------------------
  # read best_model from the metadata file saved by background code.R.
  # (metadata file records which model type had the lower test RMSE for this exact
  # predictor combination)
  
  chosen_model      <- NULL
  chosen_model_file <- NULL
  chosen_rmse       <- NA
  
  if (file.exists(meta_file)) {
    meta <- readRDS(meta_file)
    if (!is.null(meta$best_model) && !is.na(meta$best_model)) {
      chosen_model      <- meta$best_model
      chosen_rmse       <- meta$best_rmse
      chosen_model_file <- file.path(model_out_dir,
                                     paste0("mod_", combo_tag, "_", chosen_model, ".rds"))
    }
  }
  
  # load the chosen model object
  model_obj <- tryCatch(
    readRDS(chosen_model_file),
    error = function(e)
      NULL
  )
  if (is.null(model_obj)) {
    return(
      list(
        pred          = NA,
        model         = paste("Failed to load model file for:", combo_tag),
        test_rmse     = chosen_rmse,
        combo_tag     = combo_tag,
        debug_newdata = newdata
      )
    )
  }
  
  # -------------------------------------------------------------------
  # STEP 4: validate that newdata has all required columns
  # -------------------------------------------------------------------
  # extract the predictor names the loaded model actually expects
  # method differs by model class: lm stores a formula
  #                                randomForest stores predictor names in $xNames
  
  model_formula <- tryCatch(
    formula(model_obj),
    error = function(e)
      NULL
  )
  required_vars <- c()
  if (!is.null(model_formula)) {
    allvars <- all.vars(model_formula)
    if (length(allvars) >= 2)
      required_vars <- allvars[-1]  # drop response variable
  } else if (!is.null(model_obj$xNames)) {
    required_vars <- model_obj$xNames
  }
  
  missing_vars <- setdiff(required_vars, names(newdata))
  if (length(missing_vars) > 0) {
    return(
      list(
        pred          = NA,
        model         = paste(
          "Prediction failed. Missing predictors:",
          paste(missing_vars, collapse = ", ")
        ),
        test_rmse     = chosen_rmse,
        combo_tag     = combo_tag,
        debug_newdata = newdata
      )
    )
  }
  
  # -------------------------------------------------------------------
  # STEP 5: align factor levels to match training-time levels
  # -------------------------------------------------------------------
  # lm objects store the factor levels they were trained on in $xlevels
  # if newdata has a factor with a different level ordering, predict.lm()
  # throws an error
  # this loop re-levels each factor column to exactly match the stored training levels
  
  if (!is.null(model_obj$xlevels) &&
      length(model_obj$xlevels) > 0) {
    for (nm in names(model_obj$xlevels)) {
      if (nm %in% names(newdata)) {
        newdata[[nm]] <- factor(newdata[[nm]], levels = model_obj$xlevels[[nm]])
      }
    }
  }
  
  # coerce any character columns that should be numeric.
  # this can happen if a future model type stores predictor names differently
  # & a column that should be numeric arrives as character
  for (nm in names(newdata)) {
    if (is.character(newdata[[nm]]) &&
        nm %in% required_vars &&
        !(nm %in% names(model_obj$xlevels))) {
      newdata[[nm]] <- suppressWarnings(as.numeric(newdata[[nm]]))
    }
  }
  
  # -------------------------------------------------------------------
  # STEP 6: predict and back-transform
  # -------------------------------------------------------------------
  # models were trained on log(Compound_time), so the raw prediction is on
  # the log scale. exp() converts it back to minutes.
  
  pred_log <- tryCatch({
    if (inherits(model_obj, "randomForest")) {
      getS3method("predict", "randomForest")(model_obj, newdata = newdata)
    } else {
      predict(model_obj, newdata = newdata)
    }
  }, error = function(e)
    structure(NA, error_msg = conditionMessage(e)))
  
  # surface the error message if prediction failed, for easier debugging
  if (is.na(pred_log) && !is.null(attr(pred_log, "error_msg"))) {
    return(
      list(
        pred          = NA,
        model         = paste(
          "Prediction failed for combination:",
          combo_tag,
          "-",
          attr(pred_log, "error_msg")
        ),
        test_rmse     = chosen_rmse,
        combo_tag     = combo_tag,
        debug_newdata = newdata
      )
    )
  }
  if (is.na(pred_log) || length(pred_log) == 0) {
    return(
      list(
        pred          = NA,
        model         = paste("Prediction failed for combination:", combo_tag),
        test_rmse     = chosen_rmse,
        combo_tag     = combo_tag,
        debug_newdata = newdata
      )
    )
  }
  
  # back-transform from log scale to minutes
  pred_minutes <- exp(pred_log)
  
  model_label <-  switch(
    chosen_model,
    lm            = "Linear regression (lm)",
    rf            = "Random forest (rf)",
    # ADD NEW MODEL LABELS HERE ----------------------------------------------!
    
    # fallback in case you forget (or if you just don't want to bother!)
    paste(chosen_model)
  )
  
  list(
    pred          = as.numeric(pred_minutes),
    model         = model_label,
    test_rmse     = chosen_rmse,
    combo_tag     = combo_tag,
    debug_newdata = newdata
  )
}


# =============================================================================
# UI
# =============================================================================
# layout: narrow left column (width = 4) holds all inputs inside a wellPanel
#         wide right column (width = 8) displays the prediction result,
#             & internal diagnostics when Developer Mode is checked

# the UI is minimal
# Developer Mode is hidden by default & is intended to help w/ ongoing app development

ui <- fluidPage(
  # these are custom UI objects
  tags$head(tags$style(
    HTML(
      "
      /* result-box: large styled container for the predicted time */
      .result-box {
        font-size: 3rem;
        font-weight: 600;
        padding: 16px;
        border-radius: 6px;
        background: #f7fbff;
        margin-bottom: 18px;
      }
      /* help-panel: little grey instructional box at the top of the sidebar */
      .help-panel {
        background: #f8f9fa;
        padding: 10px;
        border-radius: 6px;
        margin-bottom: 12px;
      }
    "
    )
  )),
  
  titlePanel("Compounding Time Prediction Tool"),
  
  fluidRow(
    # --- LEFT COLUMN: inputs ------------------------------------------------
    column(
      width = 4,
      wellPanel(
        div(
          class = "help-panel",
          strong("How to use:"),
          p(
            "Enter any combination of inputs. Blank fields are treated as unknown."
          )
        ),
        
        # vials: must be a positive integer >= 1
        # the server auto-corrects non-integer entries after a pause
        # & throws error if <1
        div(
          title = "Must be at least 1 to make a prediction.",
          # tooltip
          numericInput(
            "vials",
            "Number of vials",
            value = NA,
            min   = 1,
            step  = 1
          )
        ),
        
        # transfer volume: must be positive, displayed to 2 decimal places
        # the server auto-corrects rounding to the nearest 0.01 mL after a pause
        # & throws error if =<0.004
        div(
          title = "Must be at least 0.01 mL to make a prediction.",
          # tooltip
          numericInput(
            "transfer",
            "Transfer volume (mL)",
            value = NA,
            min   = 0.01,
            step  = 0.01
          )
        ),
        
        # Difficulty is gated behind a checkbox because the user needs to know
        # the container type for the dose, which may not always be available
        # when unchecked, difficulty is excluded from the combo_tag entirely
        checkboxInput("use_difficulty", "Consider difficulty?", value = FALSE),
        conditionalPanel(
          condition = "input.use_difficulty == true",
          sliderInput(
            "difficulty",
            "Difficulty (1 = easy, 5 = hard)",
            min   = 1,
            max   = 5,
            value = 3,
            step  = 1
          ),
          # difficulty_desc gives a plain label for each slider value
          # so users don't need to memorize the 1-5 scale
          textOutput("difficulty_desc")
        ),
        
        # Batch: 0 = single dose, 1 = batch (multiple identical doses in sequence)
        # the blank " " choice returns NA and excludes Batch from the combo_tag
        selectInput("batch", "Batch dose?", choices = c(
          " " = NA, "No" = 0, "Yes" = 1
        )),
        
        # Facility type: maps to the Beds variable in the training data via
        # map_facility_to_beds(). blank " " returns NA
        selectInput(
          "facility_type",
          "Facility type",
          choices = c(
            " ",
            "Cancer Center",
            "Large Academic Medical Center",
            "Large Hospital",
            "Medium Hospital",
            "Small Hospital",
            "Community Hospital",
            "Women's Hospital",
            "Outpatient"
          )
        ),
        
        actionButton("go", "Predict"),
        actionButton("reset", "Reset inputs"),
        br(),
        br(),
        
        # Developer Mode: reveals combo_tag, newdata structure, chosen model,
        # RMSE, and a bar chart comparing all saved models for this combo.
        # Hidden from regular users; useful for future developers and researchers.
        checkboxInput("show_debug", "Developer Mode", value = FALSE)
      )
    ),
    
    # --- RIGHT COLUMN: outputs ----------------------------------------------
    column(
      width = 8,
      
      # primary output: predicted minutes in a large custom-styled box
      div(class = "result-box", textOutput("result")),
      br(),
      
      # Developer Mode diagnostics panel (hidden unless the checkbox is checked)
      conditionalPanel(
        condition = "input.show_debug == true",
        verbatimTextOutput("debug_combo"),
        # the combo_tag string
        verbatimTextOutput("debug_newdata"),
        # str() of the newdata row
        verbatimTextOutput("debug_model_info"),
        # chosen model name + RMSE
        plotOutput("model_compare", height = "220px"),
        # model comparison bar chart
        verbatimTextOutput("model_compare_table")       # tabular form of above
      )
    )
  )
)


# =============================================================================
# SERVER
# =============================================================================

server <- function(input, output, session) {
  # ---------------------------------------------------------------------------
  # difficulty label
  # ---------------------------------------------------------------------------
  # maps each slider position to a container type description
  # these difficulty levels come from the data dictionary used in Changfei's
  # cleaning step (see Table 1 in Changfei's capstone report)
  
  output$difficulty_desc <- renderText({
    switch(
      as.character(input$difficulty),
      "1" = "1 = syringe (easiest)",
      "2" = "2 = bag or mft vial",
      "3" = "3 = evac container or eye dropper",
      "4" = "4 = CADD or remunity cassette",
      "5" = "5 = on-Q pump (hardest)",
      ""
    )
  })
  
  # ---------------------------------------------------------------------------
  # rounding helper
  # ---------------------------------------------------------------------------
  # standard round() in R uses "banker's rounding" (round-half-to-even), which
  # can produce unexpected results at the 2-decimal boundary
  # adding a tiny epsilon before rounding gives consistent round-half-up behavior
  # for the transfer volume auto-correct display below
  
  round2 <- function(x, digits = 2) {
    eps <- 10^(-digits - 1)
    round(x + eps, digits = digits)
  }
  
  # ---------------------------------------------------------------------------
  # input auto-correction: vials
  # ---------------------------------------------------------------------------
  # debounce(..., 1500) means the correction only fires after the user pauses
  # typing for 1.5 seconds, preventing the field from updating as the user types.
  # vials must be a positive integer: we round any decimal and enforce >= 1
  
  vials_debounced <- debounce(reactive(input$vials), 1500)
  observeEvent(vials_debounced(), {
    raw <- vials_debounced()
    if (is.null(raw) || is.na(raw))
      return()
    
    tmp <- suppressWarnings(as.numeric(raw))
    if (is.na(tmp)) {
      updateNumericInput(session, "vials", value = NA)
      return()
    }
    
    if (tmp < 1)
      tmp <- 1
    rounded <- as.integer(round(tmp))
    
    # only update if the value actually changed (avoid triggering a reactive feedback loop)
    if (!isTRUE(all.equal(tmp, rounded))) {
      updateNumericInput(session, "vials", value = rounded)
    }
  })
  
  # ---------------------------------------------------------------------------
  # input auto-correction: vials
  # ---------------------------------------------------------------------------
  vials_debounced <- debounce(reactive(input$vials), 1500)
  observeEvent(vials_debounced(), {
    raw <- vials_debounced()
    if (is.null(raw) || is.na(raw))
      return()
    
    tmp <- suppressWarnings(as.numeric(raw))
    
    # if it's not a number, or if it's less than 1 (0 or negative)
    # update the UI to blank (NA) and halt this block
    if (is.na(tmp) || tmp < 1) {
      updateNumericInput(session, "vials", value = NA)
      return()
    }
    
    rounded <- as.integer(round(tmp))
    
    # only update if the value actually changed
    if (!isTRUE(all.equal(tmp, rounded))) {
      updateNumericInput(session, "vials", value = rounded)
    }
  })
  
  # ---------------------------------------------------------------------------
  # input auto-correction: transfer volume
  # ---------------------------------------------------------------------------
  transfer_debounced <- debounce(reactive(input$transfer), 1500)
  observeEvent(transfer_debounced(), {
    raw <- transfer_debounced()
    if (is.null(raw) || is.na(raw))
      return()
    
    tmp <- suppressWarnings(as.numeric(raw))
    
    # if it's not a number, or if it's 0 or negative
    # update the UI to blank (NA) and halt this block
    if (is.na(tmp) || tmp <= 0) {
      updateNumericInput(session, "transfer", value = NA)
      return()
    }
    
    rounded <- round2(tmp, 2)
    
    # only update if the value actually changed
    if (!isTRUE(all.equal(tmp, rounded))) {
      updateNumericInput(session, "transfer", value = rounded)
    }
  })
  
  # ---------------------------------------------------------------------------
  # reactive: bundle all inputs into one list
  # ---------------------------------------------------------------------------
  # centralizing all input reads here keeps the prediction logic clean &
  # makes it easier to see exactly what gets passed to predict_time()
  #
  # Batch requires extra handling -> selectInput() returns a character string
  # the blank " " choice can return either "" or "NA" depending on the browser,
  # so we guard against both before converting to numeric
  
  inputs_reactive <- reactive({
    list(
      vials    = vials_debounced(),
      transfer = transfer_debounced(),
      difficulty = if (isTRUE(input$use_difficulty))
        input$difficulty
      else
        NA,
      batch = {
        br <- input$batch
        if (!is.null(br) &&
            nzchar(as.character(br)) && !identical(as.character(br), "NA"))
          suppressWarnings(as.numeric(br))
        else
          NA
      },
      facility_type_raw = input$facility_type
    )
  })
  
  # debounce the full input bundle by 600 ms so auto-predict waits for the user
  # to pause before predicting
  inputs_debounced <- debounce(inputs_reactive, 600)
  
  # ---------------------------------------------------------------------------
  # central prediction + render function
  # ---------------------------------------------------------------------------
  # both the auto-predict observer and the manual "predict" button call this
  # same function
  # keeping the logic in one place makes them behave identially
  
  do_predict_and_render <- function(inp) {
    vials_val    <- inp$vials
    transfer_val <- inp$transfer
    
    # check whether at least one input has a usable non NA value
    has_any <- !all(is.na(
      c(
        vials_val,
        transfer_val,
        inp$difficulty,
        inp$batch,
        map_facility_to_beds(inp$facility_type_raw)
      )
    ))
    
    if (!has_any) {
      output$result              <- renderText("Enter at least one valid input to get an estimate.")
      output$debug_combo         <- renderText("")
      output$debug_newdata       <- renderText("")
      output$debug_model_info    <- renderText("")
      output$model_compare       <- renderPlot(NULL)
      output$model_compare_table <- renderText("")
      return(invisible(NULL))
    }
    
    # pass a plain list (not a reactive expression) to predict_time() so that
    # Shiny's dependency tracking doesn't propagate inside the function
    inp2 <- list(
      vials             = vials_val,
      transfer          = transfer_val,
      difficulty        = inp$difficulty,
      batch             = inp$batch,
      facility_type_raw = inp$facility_type_raw
    )
    
    result <- predict_time(inp2)
    
    # primary result: predicted minutes, or an error message
    output$result <- renderText({
      if (is.na(result$pred)) {
        "No prediction available for the selected combination."
      } else {
        paste0(round(result$pred, 4), " minutes")
      }
    })
    
    # developer diagnostics
    output$debug_combo <- renderText({
      if (is.null(result$combo_tag))
        ""
      else
        paste0("combo_tag: ", result$combo_tag)
    })
    
    # str() gives a compact structural view of the newdata row, useful for
    # confirming that transformations (log, factor levels, dummies) look right
    output$debug_newdata <- renderText({
      if (is.null(result$debug_newdata))
        ""
      else
        capture.output(str(result$debug_newdata))
    })
    
    output$debug_model_info <- renderText({
      paste0("Selected model: ",
             result$model,
             ifelse(
               is.na(result$test_rmse),
               "",
               paste0(" | test RMSE: ", round(result$test_rmse, 4))
             ))
    })
    
    # get predictions from all saved models for this combo (developer mode only)
    all_preds <- NULL
    if (!is.null(result$combo_tag) &&
        !is.null(result$debug_newdata)) {
      all_preds <- tryCatch(
        get_all_model_preds(result$combo_tag, result$debug_newdata),
        error = function(e)
          NULL
      )
    }
    
    applicable <- if (!is.null(all_preds))
      dplyr::filter(all_preds, status == "applicable")
    else
      NULL
    
    if (!is.null(applicable) && nrow(applicable) > 0) {
      # bar chart of predicted minutes per model.
      # error bars show prediction +/- test RMSE (rough indication of model uncertainty)
      output$model_compare <- renderPlot({
        df <- applicable
        df$model <- factor(df$model, levels = df$model[order(df$pred, decreasing = TRUE)])
        ggplot(df, aes(
          x = model,
          y = pred,
          fill = model
        )) +
          geom_col(show.legend = FALSE) +
          geom_errorbar(
            aes(ymin = pmax(0, pred - rmse), ymax = pred + rmse),
            width = 0.2,
            color = "gray40"
          ) +
          coord_flip() +
          ylab("Predicted minutes") +
          xlab("") +
          theme_minimal()
      })
      
      output$model_compare_table <- renderText({
        paste(capture.output(print(applicable, digits = 4)), collapse = "\n")
      })
      
    } else {
      output$model_compare       <- renderPlot(NULL)
      output$model_compare_table <- renderText("No additional models found for this input combination.")
    }
    
    invisible(result)
  }
  
  # ---------------------------------------------------------------------------
  # observers
  # ---------------------------------------------------------------------------
  
  # auto-predict: fires whenever the debounced input bundle changes
  # ignoreNULL = TRUE prevents a prediction attempt before the user has entered anything
  observeEvent(inputs_debounced(), {
    do_predict_and_render(inputs_debounced())
  }, ignoreNULL = TRUE)
  
  # manual predict: fires when the user clicks "Predict" just in case auto-predict
  # fails or is disabled (I like it for development purposes, but it might not
  # be appropriate for the end-user)
  # uses the un-debounced inputs_reactive() so the click responds immediately,
  # even if the debounce timer hasn't elapsed yet.
  observeEvent(input$go, {
    do_predict_and_render(inputs_reactive())
  })
  
  # reset: restores all inputs to their default (blank) state and clears all outputs
  # every widget that could hold a non-default value must be listed here by its input ID
  observeEvent(input$reset, {
    updateNumericInput(session, "vials", value    = NA)
    updateNumericInput(session, "transfer", value    = NA)
    updateCheckboxInput(session, "use_difficulty", value   = FALSE)
    updateSliderInput(session, "difficulty", value    = 3)
    updateSelectInput(session, "batch", selected = NA)
    updateSelectInput(session, "facility_type", selected = " ")
    updateCheckboxInput(session, "show_debug", value    = FALSE)
    
    output$result              <- renderText("")
    output$debug_combo         <- renderText("")
    output$debug_newdata       <- renderText("")
    output$debug_model_info    <- renderText("")
    output$model_compare       <- renderPlot(NULL)
    output$model_compare_table <- renderText("")
  })
}


# =============================================================================
# launch
# =============================================================================

shinyApp(ui = ui, server = server)
