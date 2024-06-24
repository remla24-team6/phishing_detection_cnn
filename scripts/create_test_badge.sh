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

# Remove existing badge line if it exists
sed '/!\[Test Status\](https:\/\/img.shields.io\/badge\/tests-/d' README.md > "$TEMP_FILE"

# Write the new badge text to the temp file
echo "$BADGE_TEXT" > README.md

# Add a newline after the badge
echo "" >> README.md

# Append the rest of the README.md content
cat "$TEMP_FILE" >> README.md

# Clean up temporary file
rm "$TEMP_FILE"

# Add and commit the updated README.md file
git add README.md
