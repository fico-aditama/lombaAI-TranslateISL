#!/bin/bash

# Check if the correct number of arguments are passed
if [ "$#" -ne 2 ]; then
    echo "Usage: $0 <file-to-split> <part-size>"
    exit 1
fi

FILE_TO_SPLIT=$1
PART_SIZE=$2

# Split the file
split -b $PART_SIZE $FILE_TO_SPLIT ${FILE_TO_SPLIT}.part-

echo "File has been split into parts."
