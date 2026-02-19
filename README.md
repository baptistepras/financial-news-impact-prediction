# NLP Tasks Project - Master 1

Academic project exploring NLP tasks on financial news data with a focus on text summarization, entity extraction and linking, and impact prediction.

students : 
- PRAS Baptiste
- PEÑA CASTAÑO Javier
- LEIVA Martin
- HERRERA NATIVI Vladimir
## Overview

This project implements a frugal pipeline for processing financial news articles using transformer-based models. The dataset consists of high-quality financial news articles annotated with Mixtral 7×8B-generated summaries and impact labels.

**Data Source**: [High-Quality Financial News Dataset](https://www.kaggle.com/datasets/sayelabualigah/high-quality-financial-news-dataset-for-nlp-tasks/data)

## Project Structure

```
.
├── dataset.csv                  # Source financial news dataset
├── train_chunk_reduce.py        # Training script with chunk-reduce approach for text summarization
├── evaluate.py                  # LLM-as-a-judge evaluation framework for text summarization
├── pred.csv                     # Model predictions output for text summarization
├── judge_results.json           # Evaluation metrics and statistics for text summarization
├── main.ipynb                   # Data exploration and entity extraction and linking
├── marketReaction.py            # Classification task for market reactions
├── outputs/                     # metrics and outputs of the calssification task
├── slides.pdf
├── report.pdf
└── README.md
```

## Tasks

### 1. Text Summarization
Fine-tuning transformer models to generate compact and detailed financial summaries using a chunk-reduce strategy for handling long documents.

### 2. Entity Extraction and Linking
Identifying and linking financial entities (companies, persons, locations, financial instruments) mentioned in news articles to external knowledge bases.

### 3. Impact Prediction
Classifying the potential market impact of financial news articles based on their content.

## Pipeline

1. **Data Loading**: Financial news articles from `dataset.csv`
2. **Preprocessing**: Tokenization and chunking for long documents
3. **Model Training**: Fine-tuning on CompactedSummary, DetailedSummary, and Impact labels
4. **Entity Processing**: Extraction and linking of financial entities
5. **Evaluation**: Multi-dimensional quality assessment using LLM-as-a-judge

## Evaluation Metrics

The evaluation framework (`evaluate.py`) assesses generated summaries across six dimensions:
- **Accuracy**: Factual correctness and attribution
- **Issuer Grounding**: Correct entity identification
- **Numeric Fidelity**: Preservation of key numbers and dates
- **Coverage**: Completeness of key events and consequences
- **Conciseness**: Information density
- **Professionalism**: Analyst-appropriate tone without filler

## Key Features

- **Chunk-reduce approach** for processing long financial documents beyond standard transformer context limits
- **LLM-based evaluation** using local instruction-tuned models for quality assessment
- **Multi-task learning** combining summarization, entity extraction, and impact prediction
- **Frugal design** optimized for academic compute constraints

## Usage

### Training
```bash
python train_chunk_reduce.py
```

### Evaluation
```bash
python evaluate.py --input_csv pred.csv --model Qwen/Qwen2.5-7B-Instruct
```

### Visualization
```bash
jupyter notebook data_vizualization.ipynb
```

## Requirements

- Python 3.8+
- PyTorch
- Transformers (Hugging Face)
- pandas, numpy
- [Additional dependencies in training/evaluation scripts]

## Academic Context

Master 1 project demonstrating practical applications of modern NLP techniques to financial domain tasks with emphasis on efficiency and interpretability.
