#!/bin/bash

# Check if the correct number of arguments are passed
if [ "$#" -ne 2 ]; then
    echo "Usage: $0 <output-file> <file-prefix>"
    exit 1
fi

OUTPUT_FILE=$1
FILE_PREFIX=$2

# Combine the parts into the output file
cat ${FILE_PREFIX}.part-* > $OUTPUT_FILE

echo "Parts have been combined into $OUTPUT_FILE."
