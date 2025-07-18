# ================================================================
# Logistic Regression for Conversion Prediction
# Dataset: Simulated Web Sessions with Timestamp
# Goal: Predict whether a session results in a conversion (1 or 0)
# Author: Joe Domaleski
# Date: [Update with current date]
# ================================================================

# ==== Step 0: Setup ====
set.seed(42)

# Load libraries
library(dplyr)
library(ggplot2)
library(lubridate)
library(caret)
library(ggpubr)

# ==== Step 1: Load Data ====
data <- read.csv("simulated_sessions_with_timestamp.csv", stringsAsFactors = TRUE)

# ==== Step 2: Feature Engineering ====
data$timestamp <- ymd_hms(data$timestamp)
data$hour <- hour(data$timestamp)

# Extract categorical variables
data$source <- as.factor(data$source)
data$converted <- as.factor(data$converted)

# ==== Step 3: Exploratory Data Analysis ====
ggplot(data, aes(x = source, fill = converted)) +
  geom_bar(position = "fill") +
  labs(title = "Conversion Rate by Source", y = "Proportion")

# ==== Step 4: Data Splitting (80/20) ====
train_index <- createDataPartition(data$converted, p = 0.8, list = FALSE)
train_data <- data[train_index, c("source", "converted")]
test_data  <- data[-train_index, c("source", "converted")]

# ==== Step 5: Fit Logistic Regression Model ====
model <- glm(converted ~ source, data = train_data, family = binomial)
summary(model)

# ==== Step 6: Predict on Test Data ====
# Predict probabilities
pred_probs <- predict(model, newdata = test_data, type = "response")

# Classify using 0.5 threshold
pred_class <- ifelse(pred_probs > 0.5, "1", "0")
pred_class <- factor(pred_class, levels = levels(test_data$converted))

# ==== Step 7: Plot of means and confidence intervals ====

# Add predicted probabilities to test data (if not already added)
test_data$predicted_prob <- predict(model, newdata = test_data, type = "response")

# Plot
ggplot(test_data, aes(x = source, y = predicted_prob, fill = source)) +
  stat_summary(fun = mean, geom = "bar", width = 0.6) +
  stat_summary(fun.data = mean_cl_normal, geom = "errorbar", width = 0.2) +
  stat_summary(geom = "text",
               fun = mean,
               aes(label = sprintf("%.3f", ..y..)),
               vjust = -0.7,
               size = 4,
               color = "black") +
  coord_cartesian(ylim = c(0.04, 0.19)) +
  labs(title = "Mean Predicted Probabilities by Source",
       x = "Source",
       y = "Predicted Probability") +
  theme_minimal() +
  theme(legend.position = "none")

# ==== Step 8: Print Log-Odds and Probabilities by Source ====
# Extract coefficients
coefs <- coef(model)

# Identify baseline (intercept) source
baseline <- "sourcead"
source_levels <- levels(train_data$source)

# Build table of log-odds and probabilities
summary_table <- data.frame(
  source = source_levels,
  log_odds = NA,
  probability = NA
)

for (i in seq_along(source_levels)) {
  level <- source_levels[i]
  if (level == "ad") {
    log_odds <- coefs["(Intercept)"]
  } else {
    coef_name <- paste0("source", level)
    log_odds <- coefs["(Intercept)"] + ifelse(coef_name %in% names(coefs), coefs[coef_name], NA)
  }
  prob <- 1 / (1 + exp(-log_odds))
  summary_table$log_odds[i] <- round(log_odds, 4)
  summary_table$probability[i] <- round(prob, 4)
}

print(summary_table)
