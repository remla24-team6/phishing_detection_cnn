#!/bin/bash

# Run tests and append output to result.log
pytest tests/ >> result.log

# Check if the test result contains "failed"
if grep -q "failed" result.log; then
    RESULT="failing"
    COLOR="red"
else
    RESULT="passing"
    COLOR="brightgreen"
fi

# Create the test status badge
BADGE_TEXT="![Test Status](https://img.shields.io/badge/tests-$RESULT-$COLOR)"

# Read metrics from reports/metrics.json
METRICS=$(cat reports/metrics.json)

# Parse metrics from JSON
TRAIN_ACCURACY=$(echo "$METRICS" | jq -r '.train_accuracy')
TRAIN_LOSS=$(echo "$METRICS" | jq -r '.train_loss')
VAL_ACCURACY=$(echo "$METRICS" | jq -r '.val_accuracy')
VAL_LOSS=$(echo "$METRICS" | jq -r '.val_loss')
TEST_ACCURACY=$(echo "$METRICS" | jq -r '.test_accuracy')
AVG_PRECISION=$(echo "$METRICS" | jq -r '.avg_precision')
AVG_RECALL=$(echo "$METRICS" | jq -r '.avg_recall')
AVG_F1=$(echo "$METRICS" | jq -r '.avg_f1')
ROC_AUC=$(echo "$METRICS" | jq -r '.roc_auc')

# Create badges for each metric
TRAIN_ACCURACY_BADGE="![Train Accuracy](https://img.shields.io/badge/train_accuracy-$TRAIN_ACCURACY-blue)"
TRAIN_LOSS_BADGE="![Train Loss](https://img.shields.io/badge/train_loss-$TRAIN_LOSS-blue)"
VAL_ACCURACY_BADGE="![Validation Accuracy](https://img.shields.io/badge/val_accuracy-$VAL_ACCURACY-blue)"
VAL_LOSS_BADGE="![Validation Loss](https://img.shields.io/badge/val_loss-$VAL_LOSS-blue)"
TEST_ACCURACY_BADGE="![Test Accuracy](https://img.shields.io/badge/test_accuracy-$TEST_ACCURACY-blue)"
AVG_PRECISION_BADGE="![Average Precision](https://img.shields.io/badge/avg_precision-$AVG_PRECISION-blue)"
AVG_RECALL_BADGE="![Average Recall](https://img.shields.io/badge/avg_recall-$AVG_RECALL-blue)"
AVG_F1_BADGE="![Average F1 Score](https://img.shields.io/badge/avg_f1-$AVG_F1-blue)"
ROC_AUC_BADGE="![ROC AUC](https://img.shields.io/badge/roc_auc-$ROC_AUC-blue)"

# Temporary file to hold the new content
TEMP_FILE=$(mktemp)

# Remove existing badges from README.md using sed
sed '/!\[Test Status\](https:\/\/img.shields.io\/badge\/tests-/d' README.md |
sed '/!\[Train Accuracy\](https:\/\/img.shields.io\/badge\/train_accuracy-/d' |
sed '/!\[Train Loss\](https:\/\/img.shields.io\/badge\/train_loss-/d' |
sed '/!\[Validation Accuracy\](https:\/\/img.shields.io\/badge\/val_accuracy-/d' |
sed '/!\[Validation Loss\](https:\/\/img.shields.io\/badge\/val_loss-/d' |
sed '/!\[Test Accuracy\](https:\/\/img.shields.io\/badge\/test_accuracy-/d' |
sed '/!\[Average Precision\](https:\/\/img.shields.io\/badge\/avg_precision-/d' |
sed '/!\[Average Recall\](https:\/\/img.shields.io\/badge\/avg_recall-/d' |
sed '/!\[Average F1 Score\](https:\/\/img.shields.io\/badge\/avg_f1-/d' |
sed '/!\[ROC AUC\](https:\/\/img.shields.io\/badge\/roc_auc-/d' > "$TEMP_FILE"


# Prepend the new badges to README.md
{
    echo "$BADGE_TEXT"
    echo "$TRAIN_ACCURACY_BADGE"
    echo "$TRAIN_LOSS_BADGE"
    echo "$VAL_ACCURACY_BADGE"
    echo "$VAL_LOSS_BADGE"
    echo "$TEST_ACCURACY_BADGE"
    echo "$AVG_PRECISION_BADGE"
    echo "$AVG_RECALL_BADGE"
    echo "$AVG_F1_BADGE"
    echo "$ROC_AUC_BADGE"
    cat "$TEMP_FILE"
} > README.md

# Clean up temporary file
rm "$TEMP_FILE"

# Add and commit the updated README.md file
git add README.md
