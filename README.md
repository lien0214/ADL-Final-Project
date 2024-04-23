# E-MoTChi: Emoji-Mandarin Translation via Computational Homophonic Intelligence

## Overview

E-MoTChi is a tuned model for emoji-to-Traditional-Mandarin translation applications.

## Key Features

1. Traditional Mandarin to Emoji (with the power of understanding homophonic)
2. Emoji to Traditional Mandarin (with the power of understanding homophonic)

## Model

Huggingface Link

## Dataset

Three different dataset under ./training_data:
```
1. 20% algo & 80% gpt
2. 50% algo & 50% gpt
3. 80% algo & 20% gpt
```

## Two Competitor model

Get into mt5 and llama folder to read README.md

## File usage
### data
This contains all the data scraped from the web, including all the sentences we want to translate and characters-to-emoji.

### data_processing
This directory contains all files used to generate the dataset.
Including web scrapers, openai api calls and codes to generate char-to-emoji

### evaluation
This directory contains scripts that is used to calculate the METEOR metric score and a script to aid human evaluation.

### llama
This directory contains all the files used to train/inference the TaiwanLLM model.
View llama/README.md for more details

### mt5
This directory contains all the files used to train/inference the google mt5 large model.
View mt5/README.md for more details

### plot
This directory contains some training graphs used in our report

### training dataset
This directory contain all the mixed training datasets and a testing file and scripts to generate the training datasets.
