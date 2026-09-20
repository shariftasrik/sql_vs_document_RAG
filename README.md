# Text-to-SQL or Document RAG?

Companion for an 80-paper IEEE-style survey of enterprise data retrieval. The repo answers **when to use SQL, document RAG, hybrid, or GraphRAG**. It does not train a new model. Reported metrics stay with their original papers and are not mixed on one leaderboard.

Layout:

| Path | Content |
|---|---|
| `Notebook/` | Colab-style notebooks |
| `code/` | Survey CLI (`run.py`, `survey.py`) and optional `sqlrag.py` |
| `paper/tex/` | IEEE sources (`main.tex`, `references.bib`, `tables/`) |
| `paper/figs/` | Plots included by the paper |
| `data/` | Coded 80-paper JSON |

## Setup

```bash
python -m venv .venv
.venv\Scripts\activate
pip install pandas numpy matplotlib
```

Linux/macOS: `source .venv/bin/activate`.

The remaining lines in `requirements.txt` are only for the optional live check in `code/sqlrag.py`.

## Rebuild the corpus (one-time after cloning extras)

```bash
python code/run.py merge
```

Merges `data/extra_papers.json` into `data/papers.json` if those 50 rows are not already present.

## Survey (tables + taxonomy + figures)

```bash
python code/run.py survey
```

Writes:

| Path | Content |
|---|---|
| `results/papers.csv` | Full coded corpus (80 rows) |
| `results/table_survey.tex` | Long supplement table |
| `results/table_mix.tex` | Count table used in the paper |
| `results/analysis.json` | Paradigm / year / failure / security counts |
| `results/taxonomy.json` | 3-tier taxonomy |
| `figs/survey_*.pdf` | Paradigm, year, taxonomy, decision, failure, loop plots |
| `paper/figs/` | Copies for Overleaf |

Last run on this tree:

- 80 papers (2015--2026)
- Text-to-SQL 28, Document RAG 24, Hybrid 18, GraphRAG 10
- Single-turn 61, self-correct 13, agent 6
- Security marked none: 76 / 80

## Decision

```bash
python code/run.py decide --q "What was total revenue in 2024?" --data tabular --has-schema --rls
python code/run.py decide --q "Summarize the main risks in last year's filings" --data text --has-docs
python code/run.py decide --q "What was net margin and why did it drop?" --data both --has-schema --has-docs
```

Prints the paradigm, the literature reason, failure modes, and which paper IDs to read.

## Corpus schema

`data/papers.json` fields:

`paper_id, title, year, venue, paradigm, primary_data_type, retrieval_mechanism, execution_engine, benchmark_used, reported_accuracy_metric, primary_failure_mode, security_privacy_support, control_loop, data_topology, retrieval_engine, use_when, avoid_when, key_innovation`

Paradigms: Text-to-SQL, Document-RAG, Hybrid-RAG-SQL, GraphRAG.

Add a paper by appending one JSON object and re-running `python code/run.py survey`.

## Decision rule (short)

1. Exact aggregate + live schema (+ RLS) → **SQL**.
2. Narrative / policy / local fact in docs → **RAG**.
3. Number whose definition sits in docs, or table+text reports → **Hybrid**.
4. Themes or multi-hop entities over a corpus → **GraphRAG**.

Spider 1.0 execution accuracy is not enterprise readiness (Spider 2.0, BEAVER, EntSQL).

## Optional live check

`code/sqlrag.py` still runs a small FinQA / T2-RAGBench experiment. It is not the survey evidence.

## License

Code is MIT (`LICENSE`). Paper metadata is our coding of public scholarly work. Underlying datasets keep their original licenses.
