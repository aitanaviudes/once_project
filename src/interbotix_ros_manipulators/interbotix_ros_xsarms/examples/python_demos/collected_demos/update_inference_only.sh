#!/bin/bash

# Define paths
BASE_ASSETS_DIR="$HOME/1000_tasks/learning_thousand_tasks/assets"
INFERENCE_DIR="$BASE_ASSETS_DIR/inference_example"

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

# Safety Check 2: Does the session directory exist?
if [ ! -d "$SESSION_NAME" ]; then
    echo "❌ Error: Source directory '$SESSION_NAME' does not exist."
    exit 1
fi

# Safety Check 3: Check for demo_0000
DEMO_PATH="$SESSION_NAME/demo_0000"
if [ ! -d "$DEMO_PATH" ]; then
    echo "❌ Error: Could not find '$DEMO_PATH'."
    exit 1
fi

echo "🚀 Preparing to update inference_example using $SESSION_NAME..."

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
    echo "Empty or missing inference directory. Proceeding to copy..."
    mkdir -p "$INFERENCE_DIR"
fi

# Step 1: Clear the contents (not the directory)
rm -rf "$INFERENCE_DIR"/*

# Step 2: Copy new files
echo "📁 Copying fresh files from $DEMO_PATH..."
cp -r "$DEMO_PATH/"* "$INFERENCE_DIR/"

echo "--------------------------------------------------"
echo "✅ Done! inference_example has been refreshed."
echo "Current contents:"
ls -1 "$INFERENCE_DIR"