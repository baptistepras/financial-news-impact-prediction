# Measuring the Market Impact of Financial News with Lightweight NLP Models

A frugal pipeline that turns raw financial news into structured `(date, ticker, impact)` events, then tests whether these events predict next-day market reactions.

Financial disclosures are long, full of numbers, and often mention several companies. Large language models summarize them well but are costly to run, and their mistakes on figures or on who did what are hard to detect. This project builds the whole chain with small models: the texts are cleaned, a fine-tuned Flan-T5 writes a short impact summary of each article, the companies are linked to their stock tickers, and each event is matched with the next-day return of the stock against its market.

## Results

**Impact summaries.** The summarizer (Flan-T5-large, about two GPU hours of fine-tuning) reaches a ROUGE-L of 0.27 against the reference summaries. ROUGE does not check figures or company names, so each summary is also audited by Qwen2.5-7B-Instruct acting as a judge, on a 0 to 5 scale:

| Accuracy | Issuer grounding | Numeric fidelity | Coverage | Conciseness | No filler |
| --- | --- | --- | --- | --- | --- |
| 3.00 | 3.17 | 3.01 | 3.07 | 2.08 | 2.38 |

These are averages over the 230 test summaries, out of 267, for which the judge returned a valid answer. The scores cluster around 3, so the judge only separates summaries coarsely. The weak points are verbosity and generic investor language.

**Market reaction.** On 259 training and 47 test events from the Saudi market, a classifier on sentence embeddings of the impact summaries reaches a test ROC-AUC of 0.45. Always predicting a positive reaction gives a higher F1 (0.79 against 0.77). With this sample size, two-sentence summaries carry no usable signal for the direction of the next-day return. The [report](docs/report.pdf) discusses richer inputs, longer windows, and magnitude targets as next steps.

## Environment

The project uses its own environment, `financial-news`, defined in [`environment.yml`](environment.yml):

- Python 3.10;
- PyTorch, Transformers, Accelerate, and Datasets for the summarizer, the judge, and the embeddings;
- fastText for language detection, rouge-score, RapidFuzz and yfinance for ticker linking and prices;
- scikit-learn, pandas, NumPy, Matplotlib, and JupyterLab.

A CUDA GPU is needed to fine-tune the summarizer and to run the judge (Qwen2.5-7B-Instruct in float16 needs about 16 GB of GPU memory). The entity linking and the market reaction steps run on a laptop, but need internet access to query Yahoo Finance. The language detection model `lid.176.bin` is downloaded separately (see [docs/usage.md](docs/usage.md)).

```bash
mamba env create -f environment.yml   # create the environment once
mamba activate financial-news         # activate it in every new terminal
```

## Data

The articles come from the [High-Quality Financial News Dataset](https://www.kaggle.com/datasets/sayelabualigah/high-quality-financial-news-dataset-for-nlp-tasks/data) on Kaggle, whose reference impact summaries were written by Mixtral 8x7B. A copy is in `data/`, together with the train and test events used for the market reaction experiment. The dataset keeps the license of its Kaggle source.

## Quick start

```bash
python summarization/train_chunk_reduce.py          # fine-tune the summarizer, write results/pred.csv
python summarization/evaluate.py                    # LLM judge, write results/judge_results.json
jupyter lab entity_linking.ipynb                    # cleaning, NER, and ticker linking
python market_reaction.py --train_csv data/train_triplets.csv --test_csv data/test_triplets.csv \
    --output_dir results/market_reaction --index_ticker "^TASI.SR"
```

Every command and its options are in [docs/usage.md](docs/usage.md).

## Repository layout

```
data/                  articles and market reaction events
summarization/         summarizer and LLM judge
entity_linking.ipynb   cleaning, NER, and ticker linking
market_reaction.py     return labels and classifier
results/               summaries, judge scores, and classifier outputs
docs/                  implementation, usage, and report
```

## Documentation

- [Implementation](docs/implementation.md): each stage of the pipeline, its models and parameters.
- [Usage](docs/usage.md): setup and every command, with its options and outputs.
- [Report](docs/report.pdf): the full write-up.

## References

- H. W. Chung, L. Hou, S. Longpre, B. Zoph, Y. Tay, W. Fedus, et al. Scaling instruction-finetuned language models. *JMLR*, 2024.
- A. Q. Jiang, A. Sablayrolles, A. Roux, A. Mensch, B. Savary, C. Bamford, et al. Mixtral of experts. arXiv:2401.04088, 2024.
- Qwen Team. Qwen2.5 technical report. arXiv:2412.15115, 2024.
- L. Zheng, W.-L. Chiang, Y. Sheng, S. Zhuang, Z. Wu, Y. Zhuang, et al. Judging LLM-as-a-judge with MT-Bench and Chatbot Arena. *NeurIPS Datasets and Benchmarks*, 2023.
- C.-Y. Lin. ROUGE: a package for automatic evaluation of summaries. *ACL Workshop on Text Summarization Branches Out*, 2004.
- A. Joulin, E. Grave, P. Bojanowski, and T. Mikolov. Bag of tricks for efficient text classification. *EACL*, 2017.
- J. Devlin, M.-W. Chang, K. Lee, and K. Toutanova. BERT: pre-training of deep bidirectional transformers for language understanding. *NAACL*, 2019.
- E. F. Tjong Kim Sang and F. De Meulder. Introduction to the CoNLL-2003 shared task: language-independent named entity recognition. *CoNLL*, 2003.
- N. Reimers and I. Gurevych. Sentence-BERT: sentence embeddings using Siamese BERT-networks. *EMNLP*, 2019.
- W. Wang, F. Wei, L. Dong, H. Bao, N. Yang, and M. Zhou. MiniLM: deep self-attention distillation for task-agnostic compression of pre-trained transformers. *NeurIPS*, 2020.

## Authors

Baptiste PRAS, Martin LEIVA, Vladimir HERRERA-NATIVI, and Javier PEÑA-CASTAÑO.
