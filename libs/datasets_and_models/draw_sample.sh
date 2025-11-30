
#!/bin/bash

# --- Configuration ---
SOURCE_DIR="$HOME/code/animals10"
DEST_DIR="$HOME/code/SaliencyAnalysisHackathonIDS/libs/datasets_and_models/sample_animals10"
SAMPLES_PER_CLASS=20  # Number of files to sample from each class

# --- Script Logic ---

# 1. Check if the source directory exists
if [ ! -d "$SOURCE_DIR" ]; then
    echo "Error: Source directory $SOURCE_DIR not found."
    exit 1
fi

# 2. Create the destination directory
mkdir -p "$DEST_DIR" || { echo "Error: Could not create destination directory $DEST_DIR"; exit 1; }

echo "Starting uniform sampling..."

# 3. Loop through each subdirectory (class) in the source directory
for CLASS_DIR in "$SOURCE_DIR"/*/; do
    # Get the class name (the last part of the path)
    CLASS_NAME=$(basename "$CLASS_DIR")
    
    # Define the destination path for the current class
    DEST_CLASS_DIR="$DEST_DIR/$CLASS_NAME"
    mkdir -p "$DEST_CLASS_DIR"

    # 4. Find all files, shuffle, select, and copy
    
    # a. Find all regular files in the class directory
    # b. Shuffle the file list randomly using 'shuf'
    # c. Select the top N files using 'head'
    # d. Read the list line by line and copy the files
    
    find "$CLASS_DIR" -maxdepth 1 -type f -print0 | shuf -z | head -z -n "$SAMPLES_PER_CLASS" | while IFS= read -r -d $'\0' FILE_PATH; do
        # Extract just the filename
        FILE_NAME=$(basename "$FILE_PATH")
        
        # Copy the file to the new destination
        cp "$FILE_PATH" "$DEST_CLASS_DIR/$FILE_NAME"
    done
    
    # Count the files copied to confirm the sample size
    COUNT=$(find "$DEST_CLASS_DIR" -maxdepth 1 -type f | wc -l)
    echo "Processed $CLASS_NAME: Copied $COUNT files to $DEST_CLASS_DIR"

done

echo "Uniform sampling complete. Sampled data is in $DEST_DIR"
