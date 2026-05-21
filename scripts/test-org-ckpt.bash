#!/bin/bash

for split in o t; do
    for fold in {0..3}; do
        python main.py --fold "${split}${fold}" --test 1 --test-bs-div 4
    done
done