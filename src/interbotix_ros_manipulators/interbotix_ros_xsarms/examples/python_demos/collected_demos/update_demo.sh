#!/bin/bash

# Define paths to keep the script clean and readable
BASE_ASSETS_DIR="$HOME/1000_tasks/learning_thousand_tasks/assets"
PICK_UP_CUBE_DIR="$BASE_ASSETS_DIR/demonstrations/pick_up_cube"
INFERENCE_DIR="$BASE_ASSETS_DIR/inference_example"

# Parse the flag from the command line (e.g., -s session_20260303_173031)
while getopts s: flag
do
    case "${flag}" in
        s) SESSION_NAME=${OPTARG};;
    esac
done

# Safety Check 1: Was a session name provided?
if [ -z "$SESSION_NAME" ]; then
    echo "Error: Please provide a session directory using the -s flag."
    echo "Usage: ./update_demo.sh -s session_20260303_173031"
    exit 1
fi

# Safety Check 2: Does the session directory exist here?
if [ ! -d "$SESSION_NAME" ]; then
    echo "Error: Directory '$SESSION_NAME' does not exist in the current folder."
    exit 1
fi

# Safety Check 3: Do the destination directories exist?
if [ ! -d "$PICK_UP_CUBE_DIR" ] || [ ! -d "$INFERENCE_DIR" ]; then
    echo "Error: One of your destination directories (pick_up_cube or inference_example) is missing."
    exit 1
fi

echo "🚀 Starting demonstration update pipeline for $SESSION_NAME..."

# Step 1: Clear out the target pick_up_cube directory
echo "--------------------------------------------------"
echo "🧹 PREVIEW: The following files in 'pick_up_cube' are about to be deleted:"
ls -1 "$PICK_UP_CUBE_DIR"
echo "--------------------------------------------------"
rm -rI "$PICK_UP_CUBE_DIR"/*

# Step 2: Copy the specified session directory into pick_up_cube
echo "📁 Copying $SESSION_NAME to pick_up_cube..."
cp -r "$SESSION_NAME" "$PICK_UP_CUBE_DIR/"

# Step 3: Move the contents of demo_0000 up into pick_up_cube
echo "📦 Extracting files from demo_0000..."
mv "$PICK_UP_CUBE_DIR/$SESSION_NAME/demo_0000/"* "$PICK_UP_CUBE_DIR/"

# Step 4: Remove the now-empty session folder
echo "--------------------------------------------------"
echo "🗑️  PREVIEW: The following leftover folder is about to be deleted:"
ls -1d "$PICK_UP_CUBE_DIR/$SESSION_NAME"
echo "--------------------------------------------------"
rm -rI "$PICK_UP_CUBE_DIR/$SESSION_NAME"

# Step 5: Update the inference_example directory
echo "--------------------------------------------------"
echo "🔄 PREVIEW: The following files in 'inference_example' are about to be deleted:"
ls -1 "$INFERENCE_DIR"
echo "--------------------------------------------------"
rm -rI "$INFERENCE_DIR"/*

echo "📁 Copying fresh files to inference_example..."
cp -r "$PICK_UP_CUBE_DIR/"* "$INFERENCE_DIR/"

echo "✅ Pipeline complete! The new demo files are ready for inference."