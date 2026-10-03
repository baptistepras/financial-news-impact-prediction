# Implementation

How each stage of the pipeline works. Commands are in [usage.md](usage.md), and the full write-up is in the [report](report.pdf).

## 1. Cleaning

Done in `summarization/train_chunk_reduce.py` for the summarizer, and in `entity_linking.ipynb` for the linking step.

- **Language.** The fastText identification model (`lid.176.bin`) keeps articles detected as English with a probability of at least 0.80.
- **Boilerplate.** Regular expressions remove forward-looking statements, safe harbor notices, tables of contents, exhibit references, and signature blocks, then whitespace is normalized.
- **Minimum length.** Very short articles and very short reference summaries are dropped.
- **Issuer.** The main company of each article is extracted with patterns such as "Company Name (NYSE: TICKER)" or "Company Name announced". It is injected into the prompts so that every summary starts with the right subject.
- **Split.** Articles are split by date: the oldest 80% for training and the most recent 20% for testing, so no future information leaks into training.

## 2. Impact summaries

`summarization/train_chunk_reduce.py` fine-tunes `google/flan-t5-large` on the reference impact summaries and generates the test summaries in two stages.

- **Chunks.** Each article is cut into overlapping chunks of 768 tokens with a stride of 384. Every chunk is paired with the summary of its article, and the loss of the first chunk is weighted by 1.35, since key figures usually come early.
- **Map.** Each chunk becomes 4 to 8 factual bullet notes: who acted, what happened, every figure, and the counterparties. The prompt asks to keep numbers and accounting terms verbatim.
- **Reduce.** The notes are merged into a 2 to 4 sentence summary that must start with "The [issuer] ...". Long documents are reduced recursively by groups of 10 notes.
- **Post-processing.** Clichés, repeated sentences, and doubled words are removed, and the issuer lead is enforced.

The best checkpoint is selected on ROUGE-L. Generation uses beam search (6 beams, length penalty 1.2). The script writes the generated summaries and their ROUGE-L scores to `results/pred.csv`.

## 3. LLM judge

`summarization/evaluate.py` asks Qwen2.5-7B-Instruct to grade each summary against its source article, from 0 to 5, on six criteria: accuracy, issuer grounding, numeric fidelity, coverage, conciseness, and absence of filler. The judge must answer in strict JSON, with the issues it found, the unsupported claims, the missing figures, and the entity confusions. The script reports the mean and the 5th and 95th percentiles of each criterion, and the best and worst examples, in `results/judge_results.json`. Answers that cannot be parsed are left out of the averages (37 of 267 in our run).

## 4. Entity extraction and ticker linking

`entity_linking.ipynb` audits the dataset, then extracts organizations from the subject and the content of each article with `dslim/bert-base-NER`, a BERT model fine-tuned on CoNLL-2003. Names are normalized and matched to Yahoo Finance tickers by fuzzy matching, and each ticker is validated. Unmatched or ambiguous names are dropped, and only tickers with enough events are kept, which leaves about 30% of the articles. Each kept article becomes a `(date, ticker, impact)` event.

## 5. Market reaction

`market_reaction.py` labels each event and trains a baseline.

- **Label.** `t0` is the first trading day on or after the announcement where both the stock and the index have a close, and `t1` is the next one. The abnormal return is the stock return minus the index return between `t0` and `t1`, and the label is 1 when it is positive. Events are restricted to Saudi tickers (suffix `.SR`) and compared with the Tadawul All Share index (`^TASI.SR`).
- **Model.** `sentence-transformers/all-MiniLM-L6-v2` embeds the Mixtral impact summary of each event, and a single linear layer is trained with a class-weighted BCE loss. The last 15% of the training period serves as validation to tune the decision threshold for F1.
- **Outputs.** Labeled events, metrics, and plots (loss, ROC curve, confusion matrix, F1 against threshold) in `results/market_reaction/`.

The classifier uses the Mixtral summaries rather than ours, so that the difficulty of predicting the reaction is measured apart from the quality of the summaries.

## Files

| File | Role |
| --- | --- |
| `data/dataset.csv` | The Kaggle articles with their Mixtral impact summaries. |
| `data/train_triplets.csv`, `data/test_triplets.csv` | `(Date, ticker, Impact)` events of the training and test periods. |
| `summarization/train_chunk_reduce.py` | Cleaning, fine-tuning, map-reduce generation, and ROUGE-L. |
| `summarization/evaluate.py` | LLM judge. |
| `entity_linking.ipynb` | Dataset audit, NER, and ticker linking. |
| `market_reaction.py` | Return labels and classifier. |
| `results/pred.csv` | Generated test summaries and their ROUGE-L. |
| `results/judge_results.json` | Judge scores, per summary and aggregated. |
| `results/market_reaction/` | Labeled events, metrics, and plots of the classifier. |
