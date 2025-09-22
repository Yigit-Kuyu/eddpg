#!/bin/bash

# EDDPG Testing Script


echo "Starting EDDPG Testing..."


SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"


if [[ "$SCRIPT_DIR" == *"/run" ]]; then
    # Script is in run folder, project root is parent directory
    PROJECT_ROOT="$(dirname "$SCRIPT_DIR")"
    MAIN_DIR="$PROJECT_ROOT/main"
else
    # Script is in project root
    PROJECT_ROOT="$SCRIPT_DIR"
    MAIN_DIR="$PROJECT_ROOT/main"
    # If main folder doesn't exist, assume files are in root
    if [ ! -d "$MAIN_DIR" ]; then
        MAIN_DIR="$PROJECT_ROOT"
    fi
fi


cd "$MAIN_DIR"

echo "Project root: $PROJECT_ROOT"
echo "Main directory: $MAIN_DIR"
echo "Current directory: $(pwd)"




echo "Make sure 'do_train: False' and 'train_resnet: False' are set in config/config.yaml"

# Add current directory to Python path and run EDDPG testing
PYTHONPATH="$MAIN_DIR:$PYTHONPATH" python3 train_test_eddpg.py

echo "EDDPG Testing completed!"
