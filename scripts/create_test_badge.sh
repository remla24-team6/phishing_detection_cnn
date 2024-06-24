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
echo "![Test Status](https://img.shields.io/badge/tests-$RESULT-$COLOR)" > badge.svg

git add badge.svg result.log