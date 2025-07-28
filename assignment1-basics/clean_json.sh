#!/bin/bash

# clean_json.sh
# Usage: ./clean_json.sh file.json

if [ $# -ne 1 ]; then
    echo "Usage: $0 file.json"
    exit 1
fi

input_file="$1"

# Create backup
cp "$input_file" "$input_file.bak"

# Remove control characters (0x00-0x1F) except tab (0x09), newline (0x0A), and carriage return (0x0D)
# Also remove 0x7F (DEL character)
LC_ALL=C tr -d '\000-\010\013-\014\016-\037\177' < "$input_file.bak" > "$input_file"

# Fix any standalone carriage returns (convert to newlines)
sed -i 's/\r/\n/g' "$input_file"

# Remove any null bytes that might have been missed
sed -i 's/\x00//g' "$input_file"

echo "Cleaned $input_file (backup saved as $input_file.bak)"

# Optional: Validate the JSON
if command -v jq &> /dev/null; then
    if jq empty "$input_file" 2>/dev/null; then
        echo "✓ JSON is valid"
        # Remove backup if successful
        rm "$input_file.bak"
    else
        echo "⚠ Warning: JSON may still have issues"
        echo "Original file saved as $input_file.bak"
        # Show where the error is
        jq empty "$input_file" 2>&1 | head -5
    fi
fi