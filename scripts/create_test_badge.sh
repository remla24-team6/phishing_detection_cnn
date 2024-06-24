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

# Create the badge
BADGE_TEXT="![Test Status](https://img.shields.io/badge/tests-$RESULT-$COLOR)" 

# Temporary file to hold the new content
TEMP_FILE=$(mktemp)

# Write the badge text to the temp file
echo "$BADGE_TEXT" > "$TEMP_FILE"

# Add a newline after the badge
echo "" >> "$TEMP_FILE"

# Append the original README.md content to the temp file
cat README.md >> "$TEMP_FILE"

# Move the temp file to README.md, effectively updating it
mv "$TEMP_FILE" README.md

git add README.md