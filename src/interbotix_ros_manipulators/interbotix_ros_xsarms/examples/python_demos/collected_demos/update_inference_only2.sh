#!/bin/bash

# Define absolute paths for consistency
BASE_ASSETS_DIR="$HOME/1000_tasks/learning_thousand_tasks/assets"
INFERENCE_DIR="$BASE_ASSETS_DIR/inference_example"
# Full path to where your session folders are actually stored
SESSION_BASE_DIR="$HOME/interbotix_ws/src/interbotix_ros_manipulators/interbotix_ros_xsarms/examples/python_demos/collected_demos"

# Parse the session name flag
while getopts s: flag
do
    case "${flag}" in
        s) SESSION_NAME=${OPTARG};;
    esac
done

# Safety Check 1: Was a session name provided?
if [ -z "$SESSION_NAME" ]; then
    echo "❌ Error: Please provide a session directory using the -s flag."
    echo "Usage: ./update_inference_only.sh -s session_20260303_173031"
    exit 1
fi

# Define the full absolute path to the data
FULL_SESSION_PATH="$SESSION_BASE_DIR/$SESSION_NAME"
DEMO_PATH="$FULL_SESSION_PATH/demo_0000"

# Safety Check 2: Does the session directory exist?
if [ ! -d "$FULL_SESSION_PATH" ]; then
    echo "❌ Error: Source directory not found at:"
    echo "   $FULL_SESSION_PATH"
    exit 1
fi

# Safety Check 3: Check for demo_0000
if [ ! -d "$DEMO_PATH" ]; then
    echo "❌ Error: Could not find demo_0000 inside the session folder."
    exit 1
fi

echo "🚀 Preparing to update inference_example using session: $SESSION_NAME"

# --- PRE-DELETION SAFETY CHECK ---
if [ -d "$INFERENCE_DIR" ] && [ "$(ls -A "$INFERENCE_DIR")" ]; then
    echo "--------------------------------------------------"
    echo "⚠️  WARNING: The following contents in 'inference_example' will be DELETED:"
    echo "--------------------------------------------------"
    # List files with details for clarity
    ls -F "$INFERENCE_DIR"
    echo "--------------------------------------------------"
    
    # Prompt user
    read -p "Press [Enter] to delete these files and proceed, or [Ctrl+C] to cancel..."
else
    echo "Inference directory is empty or missing. Creating/Proceeding..."
    mkdir -p "$INFERENCE_DIR"
fi

# Step 1: Clear the contents (not the directory)
# Using -rI here to match your preference for being asked
rm -rI "$INFERENCE_DIR"/*

# Step 2: Copy new files directly (no nesting)
echo "📁 Copying fresh files from $DEMO_PATH..."
cp -r "$DEMO_PATH/"* "$INFERENCE_DIR/"

echo "--------------------------------------------------"
echo "✅ Done! inference_example has been refreshed."
echo "Current contents of $INFERENCE_DIR:"
ls -1 "$INFERENCE_DIR"