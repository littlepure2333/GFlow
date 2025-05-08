#!/bin/bash

if [ -z "$1" ]; then
    echo "Usage: $0 tapvid_path (davis_path)"
    exit 1
fi

tapvid_path="$1"
davis_path="${2:-./data/davis}"

python utility/split_tapvid_davis.py --tapvid_path $tapvid_path --davis_path $davis_path