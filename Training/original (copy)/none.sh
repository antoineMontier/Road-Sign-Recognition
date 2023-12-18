#!/bin/bash
count=1
for file in F/* ; do
    if [ -f "$file" ]; then
        newname=$(printf "F/IMG_%04d.png" "$count")
        mv "$file" "$newname"
        ((count++))
    fi
done
