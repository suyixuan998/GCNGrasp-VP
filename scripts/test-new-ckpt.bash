#!/bin/bash

for split in o t; do
    for fold in {0..3}; do
        python main.py --test "runs/train/v2/${split}${fold}" --test-bs-div 4
    done
done