#!/bin/bash
count=1
for file in *; do
    if [ -f "$file" ]; then
        newname=$(printf "IMG_%04d.png" "$count")
        mv "$file" "$newname"
        ((count++))
    fi
done