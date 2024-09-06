Here’s a step-by-step guide to split your `sign_language_model.h5.bz2` file into smaller parts, and then how to extract and reassemble it.

### 1. Unzip `sign_language_model.h5.bz2`

To decompress the `.bz2` file:

```bash
bzip2 -d sign_language_model.h5.bz2
```

This will decompress the file and create `sign_language_model.h5`.

### 2. Split the File into Parts

You can use the `split` command to break the file into smaller parts. For example, to split the file into 40MB parts:

```bash
split -b 40M sign_language_model.h5 sign_language_model.h5.part-
```

This command will create files like `sign_language_model.h5.part-aa`, `sign_language_model.h5.part-ab`, etc.

### 3. Create a Script to Split and Extract Files

Here’s how you can create scripts to handle splitting and reassembling.

#### **Script to Split a File (split_file.sh)**

Create a file named `split_file.sh` with the following content:

```bash
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
```

Make it executable:

```bash
chmod +x split_file.sh
```

Run the script:

```bash
./split_file.sh sign_language_model.h5 40M
```

#### **Script to Combine Parts (combine_parts.sh)**

Create a file named `combine_parts.sh` with the following content:

```bash
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
```

Make it executable:

```bash
chmod +x combine_parts.sh
```

Run the script:

```bash
./combine_parts.sh sign_language_model.h5 sign_language_model.h5
```

### **Summary**

1. **Decompress the `.bz2` file**:
   ```bash
   bzip2 -d sign_language_model.h5.bz2
   ```

2. **Split the file**:
   ```bash
   split -b 40M sign_language_model.h5 sign_language_model.h5.part-
   ```

3. **Combine the parts**:
   ```bash
   cat sign_language_model.h5.part-* > sign_language_model.h5
   ```

This approach allows you to manage large files effectively by splitting them into smaller chunks and reassembling them when needed.
