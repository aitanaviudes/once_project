#!/bin/bash

# Define absolute paths
BASE_ASSETS_DIR="$HOME/1000_tasks/learning_thousand_tasks/assets"
PICK_UP_CUBE_DIR="$BASE_ASSETS_DIR/demonstrations/pick_up_cube"
INFERENCE_DIR="$BASE_ASSETS_DIR/inference_example"
# The folder where your session_... directories are stored
SESSION_BASE_DIR="$HOME/interbotix_ws/src/interbotix_ros_manipulators/interbotix_ros_xsarms/examples/python_demos/collected_demos"

# Parse the flag
while getopts s: flag
do
    case "${flag}" in
        s) SESSION_NAME=${OPTARG};;
    esac
done

# Safety Check 1: Session name provided?
if [ -z "$SESSION_NAME" ]; then
    echo "❌ Error: Please provide a session directory using the -s flag."
    exit 1
fi

# Define the source path directly to the files we want
SOURCE_DATA="$SESSION_BASE_DIR/$SESSION_NAME/demo_0000"

# Safety Check 2: Does the source data exist?
if [ ! -d "$SOURCE_DATA" ]; then
    echo "❌ Error: Could not find demo_0000 inside $SESSION_NAME."
    echo "Looking for: $SOURCE_DATA"
    exit 1
fi

echo "🚀 Starting demonstration update pipeline..."

# --- STEP 1: Update pick_up_cube ---
echo "--------------------------------------------------"
echo "🧹 Cleaning pick_up_cube contents..."
ls -F "$PICK_UP_CUBE_DIR"
# This will ask: "rm: remove all arguments recursively?"
rm -rI "$PICK_UP_CUBE_DIR"/*

echo "📁 Copying files directly to pick_up_cube..."
cp -r "$SOURCE_DATA/"* "$PICK_UP_CUBE_DIR/"

# --- STEP 2: Update inference_example ---
echo "--------------------------------------------------"
echo "🧹 Cleaning inference_example contents..."
ls -F "$INFERENCE_DIR"
# This will ask again for the second folder
rm -rI "$INFERENCE_DIR"/*

echo "📁 Copying files directly to inference_example..."
cp -r "$SOURCE_DATA/"* "$INFERENCE_DIR/"

echo "--------------------------------------------------"
echo "✅ Pipeline complete!"