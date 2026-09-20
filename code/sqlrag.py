import hashlib
import json
import math
import random
import re
import sqlite3
import time
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

import numpy as np
import pandas as pd

NUM_RE = re.compile(r"[-+]?(?:\d+\.\d+|\.\d+|\d+)(?:[eE][-+]?\d+)?")
BAD_SQL = re.compile(r"(?is)\b(insert|update|delete|drop|alter|attach|pragma|create)\b")
CACHE = Path(".cache_llm")


def load_rows(src):
    from datasets import load_dataset

    if src == "t2":
        path = "hf://datasets/G4KMU/t2-ragbench/data/FinQA/dev/metadata.jsonl"
        try:
            raw = load_dataset("json", data_files=path, split="train")
        except Exception:
            from huggingface_hub import hf_hub_download

            p = hf_hub_download(
                "G4KMU/t2-ragbench",
                "data/FinQA/dev/metadata.jsonl",
                repo_type="dataset",
            )
            raw = load_dataset("json", data_files=p, split="train")
        rows = []
        for r in raw:
            rows.append(
                {
                    "id": r["id"],
                    "doc": r["context_id"],
                    "q": r["question"],
                    "gold": r.get("program_answer") or r.get("original_answer"),
                    "table": r.get("table") or "",
                    "text": r.get("context") or "",
                }
            )
        name = "T2-RAGBench/FinQA-dev"
    else:
        raw = load_dataset("rootsautomation/FinQA", split="validation")
        rows = []
        for r in raw:
            pre = "\n".join(r["pre_text"] or [])
            post = "\n".join(r["post_text"] or [])
            tbl = r["table"] or []
            md = "\n".join(" | ".join(str(c) for c in row) for row in tbl)
            rows.append(
                {
                    "id": r["id"],
                    "doc": r["filename"] or r["id"],
                    "q": r["question"],
                    "gold": r["exe_ans"] or r["answer"],
                    "table": tbl,
                    "text": (pre + "\n" + md + "\n" + post).strip(),
                }
            )
        name = "FinQA-validation"
    return rows, name


def parse_md(md):
    out = []
    for ln in (md or "").splitlines():
        ln = ln.strip()
        if "|" not in ln:
            continue
        body = ln.replace("|", "").replace("-", "").replace(":", "").replace(" ", "")
        if body == "":
            continue
        out.append([c.strip() for c in ln.strip("|").split("|")])
    return out


def table_grid(table):
    if isinstance(table, str):
        return parse_md(table)
    return [list(r) for r in (table or [])]


def clean_name(name, i):
    s = re.sub(r"[^a-zA-Z0-9_]+", "_", str(name).strip().lower()).strip("_")
    if not s:
        s = "c%d" % i
    if s[0].isdigit():
        s = "c_" + s
    return s[:50]


def parse_cell(x):
    if x is None:
        return None
    s = str(x).strip()
    if s in ("", "-", "—", "–", "n/a", "N/A", "None", "nan"):
        return None
    s2 = s.replace(",", "").replace("$", "").replace("%", "")
    neg = len(s2) >= 2 and s2[0] == "(" and s2[-1] == ")"
    if neg:
        s2 = s2[1:-1]
    try:
        v = float(s2)
        return -v if neg else v
    except Exception:
        return s


def open_db(table):
    grid = table_grid(table)
    if len(grid) < 2:
        return None, ""
    width = max(len(r) for r in grid)
    grid = [r + [""] * (width - len(r)) for r in grid]
    seen = {}
    cols = []
    for i, name in enumerate(grid[0]):
        n = clean_name(name, i)
        if n in seen:
            seen[n] += 1
            n = "%s_%d" % (n, seen[n])
        else:
            seen[n] = 1
        cols.append(n)
    body = grid[1:]
    parsed = [[parse_cell(x) for x in r[:width]] for r in body]
    types = []
    for j in range(width):
        vals = [r[j] for r in parsed if r[j] is not None]
        n = len(vals)
        ok = sum(isinstance(v, float) for v in vals)
        types.append("real" if n and ok / n >= 0.7 else "text")
    db = sqlite3.connect(":memory:")
    db.execute("create table t (%s)" % ", ".join("%s %s" % (c, t) for c, t in zip(cols, types)))
    packed = []
    for r in parsed:
        row = []
        for j, v in enumerate(r):
            if types[j] == "real":
                row.append(v if isinstance(v, float) else None)
            else:
                row.append(None if v is None else str(v))
        packed.append(row)
    db.executemany("insert into t values (%s)" % ",".join("?" * width), packed)
    db.commit()
    sample = db.execute("select * from t limit 6").fetchall()
    schema = "table t\n" + "\n".join("- %s %s" % (c, t) for c, t in zip(cols, types))
    schema += "\nrows:\n" + "\n".join(str(x) for x in sample)
    return db, schema


def strip_sql(s):
    s = (s or "").strip()
    if s.startswith("```"):
        s = s.strip("`")
        if s[:3].lower() == "sql":
            s = s[3:]
        s = s.strip()
    if s.endswith(";"):
        s = s[:-1]
    return s.strip()


def one_select(sql):
    s = strip_sql(sql)
    if not s or ";" in s:
        return None
    if not re.match(r"(?is)^\s*select\b", s):
        return None
    if BAD_SQL.search(s):
        return None
    return s


def fmt_table(cols, data):
    if not data:
        return "(empty)"
    lines = [" | ".join(cols)]
    for r in data[:25]:
        lines.append(" | ".join("" if x is None else str(x) for x in r))
    return "\n".join(lines)


def sql_value(data):
    if not data:
        return ""
    if len(data) == 1 and len(data[0]) == 1:
        v = data[0][0]
        return "" if v is None else v
    if all(len(r) == 1 for r in data):
        vals = [r[0] for r in data if r[0] is not None]
        nums = [v for v in vals if isinstance(parse_cell(v), float)]
        if len(nums) == 1:
            return nums[0]
        return vals[0] if vals else ""
    for r in data:
        for v in r:
            if isinstance(parse_cell(v), float):
                return v
    v = data[0][0]
    return "" if v is None else v


def exec_select(db, sql):
    q = one_select(sql)
    if q is None:
        return "", "(empty)", "bad_sql"
    try:
        cur = db.execute(q)
        cols = [d[0] for d in cur.description]
        data = cur.fetchall()
    except Exception as e:
        return "", "sql err: %s" % e, "exec"
    if not data:
        return "", "(empty)", "empty"
    return sql_value(data), fmt_table(cols, data), "ok"


def gpu():
    try:
        import torch

        return "cuda" if torch.cuda.is_available() else "cpu"
    except Exception:
        return "cpu"


def qtext(q, emb):
    if "bge" in emb.lower():
        return "Represent this sentence for searching relevant passages: " + q
    return q


def build_index(rows, emb):
    from sentence_transformers import SentenceTransformer

    corpus = {}
    for r in rows:
        if r["doc"] not in corpus and r["text"]:
            corpus[r["doc"]] = r["text"]
    ids = list(corpus.keys())
    model = SentenceTransformer(emb, device=gpu())
    vecs = model.encode(
        list(corpus.values()),
        batch_size=64,
        show_progress_bar=True,
        normalize_embeddings=True,
        convert_to_numpy=True,
    )
    return {
        "ids": ids,
        "vecs": np.asarray(vecs),
        "corpus": corpus,
        "model": model,
        "id2i": {d: i for i, d in enumerate(ids)},
    }


def search_many(index, queries, k, emb):
    texts = [qtext(q, emb) for q in queries]
    qv = index["model"].encode(
        texts,
        batch_size=64,
        show_progress_bar=False,
        normalize_embeddings=True,
        convert_to_numpy=True,
    )
    S = np.asarray(qv) @ index["vecs"].T
    kk = min(k, S.shape[1])
    out = []
    for scores in S:
        part = np.argpartition(-scores, kk - 1)[:kk]
        part = part[np.argsort(-scores[part])]
        hits = []
        for j in part:
            did = index["ids"][int(j)]
            hits.append((did, float(scores[int(j)]), index["corpus"][did]))
        out.append(hits)
    return out, S


def gold_rank(index, doc, scores):
    j = index["id2i"].get(doc)
    if j is None:
        return -1
    return int(np.sum(scores > scores[j])) + 1


def make_client(key, base):
    if not key:
        return None
    from openai import OpenAI

    kw = {"api_key": key}
    if base:
        kw["base_url"] = base
    return OpenAI(**kw)


def _ckey(model, sys, user):
    h = hashlib.sha1((model + "\n" + sys + "\n" + user).encode("utf-8")).hexdigest()
    return CACHE / (h + ".json")


def ask(client, model, sys, user):
    if client is None:
        return "", 0.0, 0
    CACHE.mkdir(exist_ok=True)
    fp = _ckey(model, sys, user)
    if fp.exists():
        obj = json.loads(fp.read_text(encoding="utf-8"))
        return obj.get("text", ""), 0.0, int(obj.get("toks", 0))
    for i in range(3):
        t0 = time.perf_counter()
        try:
            r = client.chat.completions.create(
                model=model,
                temperature=0,
                max_tokens=256,
                messages=[
                    {"role": "system", "content": sys},
                    {"role": "user", "content": user},
                ],
            )
            ms = (time.perf_counter() - t0) * 1000
            text = (r.choices[0].message.content or "").strip()
            usage = getattr(r, "usage", None)
            toks = 0
            if usage is not None:
                toks = int(getattr(usage, "total_tokens", 0) or 0)
            fp.write_text(json.dumps({"text": text, "toks": toks}), encoding="utf-8")
            return text, ms, toks
        except Exception:
            time.sleep(1.2 * (i + 1))
    return "", 0.0, 0


def pred_text(s):
    s = (s or "").strip()
    if not s:
        return ""
    low = s.lower().strip().strip(".")
    if low in ("yes", "no"):
        return low
    lines = [x.strip() for x in s.splitlines() if x.strip()]
    last = lines[-1] if lines else s
    last = last.strip("`").strip()
    found = NUM_RE.findall(last.replace(",", ""))
    if len(found) == 1:
        return found[0]
    found = NUM_RE.findall(s.replace(",", ""))
    if found:
        return found[-1]
    return last


def parse_num(x):
    if x is None:
        return None
    if isinstance(x, (int, float)) and not isinstance(x, bool):
        return float(x)
    s = str(x).strip().lower()
    s = s.replace(",", "").replace("$", "").replace("%", "")
    s = re.sub(r"\s+", "", s)
    if s in ("yes", "true"):
        return "yes"
    if s in ("no", "false"):
        return "no"
    try:
        return float(s)
    except Exception:
        m = NUM_RE.search(str(x).replace(",", ""))
        if not m:
            return s
        try:
            return float(m.group(0))
        except Exception:
            return s


def em(pred, gold):
    a, b = parse_num(pred), parse_num(gold)
    if a is None or b is None or a == "" or b == "":
        return False
    if a == b:
        return True
    if isinstance(a, float) and isinstance(b, float):
        if abs(a - b) <= 1e-2:
            return True
        if abs(b) > 1e-9 and abs(a - b) / abs(b) <= 1e-2:
            return True
        for s in (100.0, 0.01, 1000.0, 0.001):
            tgt = b * s
            tol = max(1e-2, 1e-2 * abs(tgt))
            if abs(a - tgt) <= tol:
                return True
    return False


def boot_ci(xs, n=2000, seed=0):
    xs = np.asarray(xs, dtype=float)
    if len(xs) == 0:
        return 0.0, 0.0, 0.0
    rng = np.random.default_rng(seed)
    m = float(xs.mean())
    draws = rng.choice(xs, size=(n, len(xs)), replace=True).mean(axis=1)
    lo, hi = np.percentile(draws, [2.5, 97.5])
    return m, float(lo), float(hi)


def mcnemar(a, b):
    a = np.asarray(a).astype(int)
    b = np.asarray(b).astype(int)
    n01 = int(((a == 1) & (b == 0)).sum())
    n10 = int(((a == 0) & (b == 1)).sum())
    n = n01 + n10
    if n == 0:
        return n01, n10, 1.0
    if n >= 20:
        chi = (abs(n01 - n10) - 1) ** 2 / float(n)
        p = math.erfc(math.sqrt(chi / 2.0))
        return n01, n10, float(min(1.0, p))
    p = 0.0
    for i in range(n + 1):
        if abs(i - n / 2.0) >= abs(n01 - n / 2.0):
            p += math.comb(n, i) * (0.5 ** n)
    return n01, n10, float(min(1.0, p))


def sql_tag(ok, kind):
    if ok:
        return "ok"
    if kind in ("bad_sql", "no_table", "no_sql"):
        return kind
    if kind in ("exec", "empty"):
        return kind
    return "wrong"


def rag_tag(ok, hit, pred):
    if ok:
        return "ok"
    if not pred:
        return "empty"
    if not hit:
        return "miss"
    return "wrong"


def sql_answer(row, client, model):
    db, schema = open_db(row["table"])
    if db is None:
        return "", "(empty)", "", "no_table", 0.0, 0
    txt, ms, toks = ask(
        client,
        model,
        "Write one SQLite SELECT over table t. Return SQL only.",
        "Schema:\n%s\n\nQuestion: %s" % (schema, row["q"]),
    )
    pred, table, kind = exec_select(db, txt)
    db.close()
    return pred, table, strip_sql(txt), kind, ms, toks


def rag_answer(row, hits, client, model):
    chunks = []
    for i, (doc, sc, text) in enumerate(hits):
        chunks.append("[%d] %s (%.3f)\n%s" % (i + 1, doc, sc, text[:1800]))
    txt, ms, toks = ask(
        client,
        model,
        "Answer from the passages. If the answer is a number, output only that number.",
        "Question: %s\n\n%s" % (row["q"], "\n\n".join(chunks)),
    )
    return pred_text(txt), ms, toks, txt


def hyb_answer(row, table, hits, client, model):
    chunks = []
    for i, (doc, sc, text) in enumerate(hits[:3]):
        chunks.append("[%d] %s\n%s" % (i + 1, doc, text[:1200]))
    txt, ms, toks = ask(
        client,
        model,
        "Use the SQL result first. Use passages only to fill gaps. Output only the final answer.",
        "Question: %s\n\nSQL result:\n%s\n\nPassages:\n%s"
        % (row["q"], table, "\n\n".join(chunks)),
    )
    return pred_text(txt), ms, toks, txt


def pick(rows, n, seed):
    rng = random.Random(seed)
    order = list(range(len(rows)))
    rng.shuffle(order)
    if n is None or n <= 0 or n > len(order):
        order_n = order
    else:
        order_n = order[:n]
    return [rows[i] for i in order_n]


def run_eval(rows, cfg, client):
    sample = pick(rows, cfg["n"], cfg["seed"])
    t0 = time.perf_counter()
    index = build_index(rows, cfg["emb"])
    hits_pack, S = search_many(index, [r["q"] for r in sample], cfg["k"], cfg["emb"])
    ret_ms = (time.perf_counter() - t0) * 1000
    out = []
    pool = ThreadPoolExecutor(max_workers=2) if client else None
    try:
        for i, row in enumerate(sample):
            hits = hits_pack[i]
            scores = S[i]
            rank = gold_rank(index, row["doc"], scores)
            hit = int(rank != -1 and rank <= cfg["k"])
            if client is None:
                rec = {
                    "id": row["id"],
                    "q": row["q"],
                    "gold": row["gold"],
                    "sql": "",
                    "rag": "",
                    "hyb": "",
                    "sql_ok": 0,
                    "rag_ok": 0,
                    "hyb_ok": 0,
                    "hit": hit,
                    "rank": rank,
                    "sql_tag": "no_sql",
                    "rag_tag": "empty",
                    "hyb_tag": "empty",
                    "query": "",
                    "ms_sql": 0.0,
                    "ms_rag": 0.0,
                    "ms_hyb": 0.0,
                    "tok_sql": 0,
                    "tok_rag": 0,
                    "tok_hyb": 0,
                }
                out.append(rec)
                continue

            f_sql = pool.submit(sql_answer, row, client, cfg["model"])
            f_rag = pool.submit(rag_answer, row, hits, client, cfg["model"])
            sp, table, sql, kind, ms_s, tok_s = f_sql.result()
            rp, ms_r, tok_r, _raw_r = f_rag.result()
            hp, ms_h, tok_h, _raw_h = hyb_answer(row, table, hits, client, cfg["model"])
            sok = int(em(sp, row["gold"]))
            rok = int(em(rp, row["gold"]))
            hok = int(em(hp, row["gold"]))
            rec = {
                "id": row["id"],
                "q": row["q"],
                "gold": row["gold"],
                "sql": "" if sp is None else str(sp),
                "rag": rp,
                "hyb": hp,
                "sql_ok": sok,
                "rag_ok": rok,
                "hyb_ok": hok,
                "hit": hit,
                "rank": rank,
                "sql_tag": sql_tag(sok, kind),
                "rag_tag": rag_tag(rok, hit, rp),
                "hyb_tag": rag_tag(hok, hit, hp),
                "query": sql,
                "ms_sql": ms_s,
                "ms_rag": ms_r,
                "ms_hyb": ms_h,
                "tok_sql": tok_s,
                "tok_rag": tok_r,
                "tok_hyb": tok_h,
            }
            out.append(rec)
            if (i + 1) % 5 == 0 or i == 0:
                print(i + 1, "/", len(sample), "gold=", row["gold"], "sql=", rec["sql"], "rag=", rp)
    finally:
        if pool is not None:
            pool.shutdown(wait=True)
    df = pd.DataFrame(out)
    meta = {
        "n": int(len(df)),
        "pool": int(len(rows)),
        "src": cfg["src"],
        "dataset": cfg["dataset"],
        "model": cfg["model"],
        "emb": cfg["emb"],
        "k": cfg["k"],
        "seed": cfg["seed"],
        "index_ms": ret_ms,
        "has_llm": bool(client),
        "versions": _versions(),
    }
    return df, meta


def _versions():
    out = {}
    for n in ("datasets", "sentence_transformers", "openai", "pandas", "numpy", "matplotlib"):
        try:
            m = __import__(n)
            out[n] = getattr(m, "__version__", "?")
        except Exception:
            out[n] = None
    return out


def summarize(df):
    rows = []
    for key, col in (("SQL", "sql_ok"), ("RAG", "rag_ok"), ("Hybrid", "hyb_ok")):
        m, lo, hi = boot_ci(df[col].values)
        rows.append({"method": key, "acc": m, "lo": lo, "hi": hi})
    rec = pd.DataFrame(rows)
    rec.loc[len(rec)] = {
        "method": "R@k",
        "acc": float(df["hit"].mean()) if len(df) else 0.0,
        "lo": boot_ci(df["hit"].values)[1],
        "hi": boot_ci(df["hit"].values)[2],
    }
    tests = {}
    pairs = (("SQL", "RAG", "sql_ok", "rag_ok"), ("SQL", "Hybrid", "sql_ok", "hyb_ok"), ("RAG", "Hybrid", "rag_ok", "hyb_ok"))
    for a, b, ca, cb in pairs:
        n01, n10, p = mcnemar(df[ca].values, df[cb].values)
        tests["%s_vs_%s" % (a, b)] = {"a_win": n01, "b_win": n10, "p": p}
    return rec, tests


C_SQL = "#0072B2"
C_RAG = "#E69F00"
C_HYB = "#009E73"
C_OTH = "#6B6B6B"
COLS = {"SQL": C_SQL, "RAG": C_RAG, "Hybrid": C_HYB}


def _mpl():
    import matplotlib.pyplot as plt

    plt.rcParams.update(
        {
            "font.family": "DejaVu Sans",
            "font.size": 10,
            "axes.titlesize": 11,
            "axes.labelsize": 10,
            "figure.dpi": 120,
            "savefig.dpi": 300,
            "pdf.fonttype": 42,
            "ps.fonttype": 42,
            "axes.spines.top": False,
            "axes.spines.right": False,
        }
    )
    return plt


def _save(fig, folder, name, plt, rect=None):
    folder.mkdir(parents=True, exist_ok=True)
    if rect is None:
        fig.tight_layout()
    else:
        fig.tight_layout(rect=rect)
    pdf = folder / (name + ".pdf")
    png = folder / (name + ".png")
    fig.savefig(pdf, bbox_inches="tight")
    fig.savefig(png, bbox_inches="tight")
    plt.close(fig)
    return [str(pdf), str(png)]


def _bars(ax, rec):
    rec = rec[rec["method"] != "R@k"]
    x = np.arange(len(rec))
    y = rec["acc"].values
    lo = np.minimum(rec["lo"].values, y)
    hi = np.maximum(rec["hi"].values, y)
    yerr = np.vstack([y - lo, hi - y])
    colors = [COLS[m] for m in rec["method"]]
    ax.bar(x, y, color=colors, width=0.62, yerr=yerr, capsize=3, ecolor="#222222", error_kw={"linewidth": 1})
    ax.set_xticks(x)
    ax.set_xticklabels(list(rec["method"]))
    ax.set_ylim(0, 1)
    ax.set_ylabel("Execution EM")
    ax.set_title("(a) Accuracy (95% bootstrap CI)")
    for i, v in enumerate(y):
        ax.text(i, min(0.97, v + 0.04), "%.2f" % v, ha="center", va="bottom", fontsize=8)


def _fail(ax, df):
    order = ["ok", "wrong", "miss", "exec", "empty", "bad_sql", "no_table", "no_sql"]
    methods = [("SQL", "sql_tag"), ("RAG", "rag_tag"), ("Hybrid", "hyb_tag")]
    palette = {
        "ok": C_HYB,
        "wrong": "#D55E00",
        "miss": "#CC79A7",
        "exec": "#56B4E9",
        "empty": "#999999",
        "bad_sql": "#0072B2",
        "no_table": "#666666",
        "no_sql": "#444444",
    }
    bottom = np.zeros(3)
    n = max(len(df), 1)
    used = []
    for tag in order:
        vals = []
        for _, col in methods:
            vals.append(float((df[col] == tag).sum()) / n)
        if sum(vals) == 0:
            continue
        ax.bar(np.arange(3), vals, bottom=bottom, color=palette[tag], width=0.62, label=tag)
        bottom = bottom + np.array(vals)
        used.append(tag)
    ax.set_xticks(np.arange(3))
    ax.set_xticklabels(["SQL", "RAG", "Hybrid"])
    ax.set_ylim(0, 1)
    ax.set_ylabel("Share")
    ax.set_title("(b) Failure taxonomy")
    if used:
        ax.legend(frameon=False, fontsize=7, loc="upper right")


def _lat(ax, df):
    data = [df["ms_sql"].values, df["ms_rag"].values, df["ms_hyb"].values]
    bp = ax.boxplot(data, labels=["SQL", "RAG", "Hybrid"], patch_artist=True, widths=0.55, showfliers=False)
    for patch, c in zip(bp["boxes"], [C_SQL, C_RAG, C_HYB]):
        patch.set_facecolor(c)
        patch.set_alpha(0.75)
        patch.set_linewidth(0.8)
    for w in bp["whiskers"] + bp["caps"]:
        w.set_color("#222222")
    for med in bp["medians"]:
        med.set_color("#111111")
    ax.set_ylabel("Latency (ms)")
    ax.set_title("(c) LLM latency")


def _overlap(ax, df):
    s = df["sql_ok"].astype(int).values
    r = df["rag_ok"].astype(int).values
    h = df["hyb_ok"].astype(int).values
    labs = ["Both", "SQL only", "RAG only", "Neither", "Hybrid rescue"]
    vals = [
        float(((s == 1) & (r == 1)).mean()),
        float(((s == 1) & (r == 0)).mean()),
        float(((s == 0) & (r == 1)).mean()),
        float(((s == 0) & (r == 0)).mean()),
        float(((h == 1) & (s == 0) & (r == 0)).mean()),
    ]
    colors = [C_HYB, C_SQL, C_RAG, C_OTH, "#56B4E9"]
    ax.bar(np.arange(5), vals, color=colors, width=0.7)
    ax.set_xticks(np.arange(5))
    ax.set_xticklabels(labs, rotation=20, ha="right")
    ax.set_ylim(0, 1)
    ax.set_ylabel("Share")
    ax.set_title("(d) Complementarity")


def _hit(ax, df):
    xs = []
    h_rag = []
    h_hyb = []
    for v in (0, 1):
        part = df[df["hit"] == v]
        xs.append("miss" if v == 0 else "hit")
        if part.empty:
            h_rag.append(0.0)
            h_hyb.append(0.0)
            continue
        h_rag.append(float(part["rag_ok"].mean()))
        h_hyb.append(float(part["hyb_ok"].mean()))
    x = np.arange(len(xs))
    ax.bar(x - 0.16, h_rag, 0.32, color=C_RAG, label="RAG")
    ax.bar(x + 0.16, h_hyb, 0.32, color=C_HYB, label="Hybrid")
    ax.set_xticks(x)
    ax.set_xticklabels(xs)
    ax.set_ylim(0, 1)
    ax.set_ylabel("EM")
    ax.set_title("Accuracy by retrieval hit")
    ax.legend(frameon=False, fontsize=8)


def draw_figs(df, meta, out="figs"):
    plt = _mpl()
    folder = Path(out)
    paths = []
    rec, _tests = summarize(df)

    fig, axes = plt.subplots(2, 2, figsize=(9.4, 7.0))
    _bars(axes[0, 0], rec)
    _fail(axes[0, 1], df)
    _lat(axes[1, 0], df)
    _overlap(axes[1, 1], df)
    fig.suptitle("%s  n=%d  %s" % (meta.get("dataset", ""), meta.get("n", 0), meta.get("model", "")), fontsize=11)
    paths += _save(fig, folder, "fig_main", plt, rect=(0, 0, 1, 0.96))

    fig2, ax = plt.subplots(figsize=(4.6, 3.4))
    _hit(ax, df)
    paths += _save(fig2, folder, "fig_retrieval", plt)

    fig3, ax = plt.subplots(figsize=(4.6, 3.4))
    _bars(ax, rec)
    ax.set_title("Accuracy (95% CI)")
    paths += _save(fig3, folder, "fig_acc", plt)

    fig4, ax = plt.subplots(figsize=(5.2, 3.6))
    _fail(ax, df)
    ax.set_title("Failure taxonomy")
    paths += _save(fig4, folder, "fig_fail", plt)

    return paths


def save_run(df, meta, rec, tests, out="results"):
    folder = Path(out)
    folder.mkdir(parents=True, exist_ok=True)
    df.to_csv(folder / "predictions.csv", index=False)
    rec.to_csv(folder / "summary.csv", index=False)
    payload = dict(meta)
    payload["tests"] = tests
    payload["summary"] = rec.to_dict(orient="records")
    (folder / "run.json").write_text(json.dumps(payload, indent=2), encoding="utf-8")
    return str(folder / "predictions.csv"), str(folder / "run.json")
