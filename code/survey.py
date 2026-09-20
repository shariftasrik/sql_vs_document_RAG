import json
import re
from collections import Counter, defaultdict
from pathlib import Path

import matplotlib.pyplot as plt
import pandas as pd

CODE_DIR = Path(__file__).resolve().parent
ROOT = CODE_DIR.parent
PAPERS = ROOT / "data" / "papers.json"
EXTRA = ROOT / "data" / "extra_papers.json"
MIN_PAPERS = 80
REQUIRED = (
    "paper_id",
    "title",
    "year",
    "venue",
    "paradigm",
    "benchmark_used",
    "reported_accuracy_metric",
    "primary_failure_mode",
    "security_privacy_support",
    "control_loop",
    "data_topology",
    "retrieval_engine",
)

AGG_RE = re.compile(
    r"\b(sum|total|average|avg|mean|count|how many|how much|max|min|median|"
    r"percentage|percent|ratio|growth|yoy|revenue|gmv|arr)\b",
    re.I,
)
SEM_RE = re.compile(
    r"\b(why|explain|summar|theme|policy|opinion|risks?|narrative|"
    r"what are the main|discuss|describe)\b",
    re.I,
)
JOIN_RE = re.compile(r"\b(join|across (tables|systems)|per (customer|sku|region))\b", re.I)

RULES = [
    {
        "id": "sql",
        "label": "Text-to-SQL",
        "when": "Typed columns exist and the answer is an exact aggregate, filter, or join.",
        "not_when": "The fact lives only in PDFs, or the user wants a theme/summary.",
        "watch": "Wrong JOIN, schema mis-link, empty set, Spider-style overconfidence.",
        "paradigms": ["Text-to-SQL"],
    },
    {
        "id": "rag",
        "label": "Document RAG",
        "when": "Unstructured docs; the answer is a span, policy, or local fact.",
        "not_when": "You need SUM/COUNT that must match the warehouse.",
        "watch": "Hallucination, chunk miss, lost-in-the-middle.",
        "paradigms": ["Document-RAG"],
    },
    {
        "id": "hybrid",
        "label": "Hybrid (SQL-first + RAG)",
        "when": "A number comes from tables and the wording/metric definition comes from docs.",
        "not_when": "One side is missing (no schema, or no documents).",
        "watch": "Bad evidence then wrong SQL (EntSQL); retrieval miss then wrong arithmetic (TAT-QA).",
        "paradigms": ["Hybrid-RAG-SQL"],
    },
    {
        "id": "graph",
        "label": "GraphRAG",
        "when": "Corpus-level themes or multi-hop entities across many documents.",
        "not_when": "Exact KPI over a star schema.",
        "watch": "Index cost; graph errors; weak numeric fidelity.",
        "paradigms": ["GraphRAG"],
    },
]


def load_papers(path=None):
    p = Path(path) if path else PAPERS
    rows = json.loads(p.read_text(encoding="utf-8"))
    if not isinstance(rows, list):
        raise ValueError("papers file must be a JSON list")
    ids = [r.get("paper_id") for r in rows]
    if len(ids) != len(set(ids)):
        raise ValueError("duplicate paper_id in corpus")
    missing = [r.get("paper_id") for r in rows if any(k not in r for k in REQUIRED)]
    if missing:
        raise ValueError("missing fields in: " + ", ".join(str(x) for x in missing[:8]))
    if len(rows) < MIN_PAPERS:
        raise ValueError("need at least %d papers, found %d" % (MIN_PAPERS, len(rows)))
    return rows


def merge_corpus(core=None, extra=None, dest=None):
    core = Path(core) if core else PAPERS
    extra = Path(extra) if extra else EXTRA
    dest = Path(dest) if dest else PAPERS
    a = json.loads(core.read_text(encoding="utf-8"))
    b = json.loads(extra.read_text(encoding="utf-8")) if extra.exists() else []
    seen = {r["paper_id"] for r in a}
    for r in b:
        if r["paper_id"] in seen:
            continue
        a.append(r)
        seen.add(r["paper_id"])
    dest.write_text(json.dumps(a, indent=2) + "\n", encoding="utf-8")
    return a


def frame(rows):
    return pd.DataFrame(rows)


def taxonomy(rows):
    tree = defaultdict(lambda: defaultdict(lambda: defaultdict(list)))
    for r in rows:
        t1 = r.get("data_topology") or "Unknown"
        t2 = r.get("retrieval_engine") or "Unknown"
        t3 = r.get("control_loop") or "Unknown"
        tree[t1][t2][t3].append(r["paper_id"])
    return tree


def print_taxonomy(tree):
    print("TIER 1 Data topology")
    for t1, a in sorted(tree.items()):
        n1 = sum(len(v) for b in a.values() for v in b.values())
        print("  %s (%d)" % (t1, n1))
        print("  TIER 2 Retrieval engine")
        for t2, b in sorted(a.items()):
            n2 = sum(len(v) for v in b.values())
            print("    %s (%d)" % (t2, n2))
            print("    TIER 3 Control loop")
            for t3, ids in sorted(b.items()):
                print("      %s: %s" % (t3, ", ".join(ids)))


def latex_table(rows, path):
    cols = [
        ("paper_id", "Paper ID"),
        ("paradigm", "Paradigm"),
        ("data_topology", "Data Topology"),
        ("benchmark_used", "Primary Benchmark"),
        ("reported_accuracy_metric", "Reported Metric"),
        ("primary_failure_mode", "Primary Failure Mode"),
        ("security_privacy_support", "Security"),
    ]
    lines = [
        r"\begin{table*}[t]",
        r"\centering",
        r"\small",
        r"\caption{Full coded sheet (supplement). Metrics stay with the paper that published them and are not ranked across paradigms.}",
        r"\label{tab:survey}",
        r"\begin{tabular}{p{2.4cm}p{2.2cm}p{1.8cm}p{2.4cm}p{3.4cm}p{3.2cm}p{2.0cm}}",
        r"\toprule",
        " & ".join(c[1] for c in cols) + r" \\",
        r"\midrule",
    ]
    for r in rows:
        cells = []
        for k, _ in cols:
            s = str(r.get(k, "")).replace("&", r"\&").replace("%", r"\%").replace("_", r"\_")
            s = s.replace("\n", " ")
            if len(s) > 90:
                s = s[:87] + "..."
            cells.append(s)
        lines.append(" & ".join(cells) + r" \\")
    lines += [r"\bottomrule", r"\end{tabular}", r"\end{table*}", ""]
    Path(path).write_text("\n".join(lines), encoding="utf-8")
    return path


def score_query(q, data, has_schema, has_docs, rls):
    q = q or ""
    agg = bool(AGG_RE.search(q))
    sem = bool(SEM_RE.search(q))
    join = bool(JOIN_RE.search(q))
    data = (data or "auto").lower()
    if data == "auto":
        if has_schema and not has_docs:
            data = "tabular"
        elif has_docs and not has_schema:
            data = "text"
        elif has_schema and has_docs:
            data = "both"
        else:
            data = "both" if (agg and sem) else ("tabular" if agg else "text")
    return {
        "agg": agg or data == "tabular",
        "sem": sem or data == "text",
        "join": join,
        "data": data,
        "has_schema": has_schema or data in ("tabular", "both"),
        "has_docs": has_docs or data in ("text", "both"),
        "rls": rls,
    }


def decide(rows, q="", data="auto", has_schema=False, has_docs=False, rls=False):
    f = score_query(q, data, has_schema, has_docs, rls)
    if f["rls"] and f["has_schema"] and f["agg"] and not f["sem"]:
        pick = "sql"
    elif f["data"] == "tabular" and f["agg"] and not f["sem"]:
        pick = "sql"
    elif f["data"] == "text" and not f["agg"]:
        pick = "rag"
    elif f["sem"] and not f["agg"] and not f["has_schema"]:
        pick = "graph" if re.search(r"\b(theme|corpus|across (all|the) (docs|filings))\b", q or "", re.I) else "rag"
    elif f["has_schema"] and f["has_docs"]:
        pick = "hybrid"
    elif f["agg"] and f["sem"]:
        pick = "hybrid"
    elif f["agg"]:
        pick = "sql"
    else:
        pick = "rag"

    rule = next(r for r in RULES if r["id"] == pick)
    support = sorted(
        [p for p in rows if p["paradigm"] in rule["paradigms"]],
        key=lambda x: int(x.get("year") or 0),
        reverse=True,
    )
    contra = sorted(
        [p for p in rows if p["paradigm"] not in rule["paradigms"]],
        key=lambda x: int(x.get("year") or 0),
        reverse=True,
    )
    return {
        "choice": rule["label"],
        "rule_id": pick,
        "because": rule["when"],
        "not_when": rule["not_when"],
        "watch": rule["watch"],
        "features": f,
        "read_first": [p["paper_id"] for p in support[:8]],
        "contrast": [p["paper_id"] for p in contra[:6]],
        "use_when": [p["use_when"] for p in support[:5]],
    }


def print_decision(d):
    print("USE:", d["choice"])
    print("because:", d["because"])
    print("do not use when:", d["not_when"])
    print("failure modes:", d["watch"])
    print("read:", ", ".join(d["read_first"]))
    print("contrast:", ", ".join(d["contrast"]))
    print("features:", d["features"])


def _sec_none(r):
    s = str(r.get("security_privacy_support") or "").strip().lower()
    return s in ("", "none", "n/a", "no")


def failure_family(r):
    t = (r.get("primary_failure_mode") or "").lower()
    if any(k in t for k in ("rls", "isolat", "tenant", "not a real database", "warehouse calculator")):
        return "isolation_gap"
    if any(k in t for k in ("join", "grain", "cartesian", "group by", "nested", "scale", "unit", "million", "aggregation head")):
        return "join_grain"
    if any(k in t for k in ("schema", "link", "column", "table name", "ugly name", "header", "sub-table", "synonym", "perturb")):
        return "schema_link"
    if any(k in t for k in ("syntax", "illegal", "parse", "unexecutable", "api call", "compile")):
        return "syntax"
    if any(k in t for k in ("hallucin", "invent", "unsupported", "faithful", "counterfactual", "noise", "unfaithful")):
        return "hallucination"
    if any(k in t for k in ("lost-in-the-middle", "middle", "long context", "window", "truncat", "layout", "pdf")):
        return "long_context"
    if any(k in t for k in ("graph", "entity", "subgraph", "pagerank", "community", "steiner", "openie", "kg")):
        return "graph_index"
    if any(k in t for k in ("prompt", "few-shot", "demonstration", "candidate", "ranker")):
        return "prompting"
    if any(k in t for k in ("turn", "dialogue", "clarification", "chain across")):
        return "dialogue_state"
    if any(k in t for k in ("dialect", "workflow", "multi-query", "project file")):
        return "workflow"
    if any(k in t for k in ("judge", "metric", "not a substitute", "not ex")):
        return "evaluation"
    if any(k in t for k in ("evidence", "document", "policy", "knowledge", "hop", "retriev", "chunk", "page", "stale", "domain shift")):
        return "evidence_miss"
    return "other"


def counts(rows):
    return {
        "n": len(rows),
        "paradigm": dict(Counter(r["paradigm"] for r in rows)),
        "topology": dict(Counter(r["data_topology"] for r in rows)),
        "loop": dict(Counter(r["control_loop"] for r in rows)),
        "engine": dict(Counter(r["retrieval_engine"] for r in rows)),
        "year": dict(Counter(int(r["year"]) for r in rows)),
        "security_none": sum(1 for r in rows if _sec_none(r)),
        "security_mentioned": sum(1 for r in rows if not _sec_none(r)),
        "failure_family": dict(Counter(failure_family(r) for r in rows)),
        "arxiv": sum(1 for r in rows if "arxiv" in str(r.get("venue", "")).lower()),
        "year_min": min(int(r["year"]) for r in rows),
        "year_max": max(int(r["year"]) for r in rows),
    }


def analysis(rows):
    c = counts(rows)
    df = frame(rows)
    by_para = {}
    for p, sub in df.groupby("paradigm"):
        by_para[p] = {
            "n": int(len(sub)),
            "share": round(len(sub) / len(df), 4),
            "single_turn": int((sub["control_loop"] == "Single-turn").sum()),
            "self_correct": int((sub["control_loop"] == "Self-Correction").sum()),
            "agent": int((sub["control_loop"] == "Agentic Routing").sum()),
            "sec_none": int(sum(_sec_none(r) for r in sub.to_dict("records"))),
        }
    era = {
        "pre_2020": int((df["year"] < 2020).sum()),
        "2020_2022": int(((df["year"] >= 2020) & (df["year"] <= 2022)).sum()),
        "2023_2024": int(((df["year"] >= 2023) & (df["year"] <= 2024)).sum()),
        "2025_2026": int((df["year"] >= 2025).sum()),
    }
    c["by_paradigm"] = by_para
    c["era"] = era
    c["topology_loop"] = pd.crosstab(df["data_topology"], df["control_loop"]).to_dict()
    return c


def latex_mix(rows, path):
    c = counts(rows)
    lines = [
        r"\begin{table}[!t]",
        r"\caption{Coded corpus used in this survey.}",
        r"\label{tab:mix}",
        r"\centering",
        r"\begin{tabular}{lc}",
        r"\toprule",
        r"Slice & Count \\",
        r"\midrule",
        r"Papers & %d \\" % c["n"],
        r"Text-to-SQL & %d \\" % c["paradigm"].get("Text-to-SQL", 0),
        r"Document RAG & %d \\" % c["paradigm"].get("Document-RAG", 0),
        r"Hybrid (table-text or schema+SQL) & %d \\" % c["paradigm"].get("Hybrid-RAG-SQL", 0),
        r"GraphRAG & %d \\" % c["paradigm"].get("GraphRAG", 0),
        r"Tabular / unstructured / hybrid topology & %d / %d / %d \\"
        % (
            c["topology"].get("Tabular", 0),
            c["topology"].get("Unstructured", 0),
            c["topology"].get("Hybrid", 0),
        ),
        r"Single turn / self-correct / agent route & %d / %d / %d \\"
        % (
            c["loop"].get("Single-turn", 0),
            c["loop"].get("Self-Correction", 0),
            c["loop"].get("Agentic Routing", 0),
        ),
        r"Security / isolation discussed & %d \\" % c["security_mentioned"],
        r"Security marked none & %d \\" % c["security_none"],
        r"\bottomrule",
        r"\end{tabular}",
        r"\end{table}",
        "",
    ]
    Path(path).write_text("\n".join(lines), encoding="utf-8")
    return path


def style():
    plt.rcParams.update(
        {
            "font.family": "DejaVu Sans",
            "font.size": 10,
            "axes.spines.top": False,
            "axes.spines.right": False,
            "figure.dpi": 120,
            "savefig.dpi": 300,
            "pdf.fonttype": 42,
        }
    )


def draw(rows, out="figs"):
    style()
    folder = Path(out)
    folder.mkdir(parents=True, exist_ok=True)
    df = frame(rows)
    paths = []

    fig, ax = plt.subplots(figsize=(6.2, 3.6))
    vc = df["paradigm"].value_counts()
    ax.barh(list(vc.index)[::-1], list(vc.values)[::-1], color="#0072B2")
    ax.set_xlabel("Papers")
    ax.set_title("Survey corpus by paradigm (n=%d)" % len(df))
    fig.tight_layout()
    p = folder / "survey_paradigm"
    fig.savefig(str(p) + ".pdf", bbox_inches="tight")
    fig.savefig(str(p) + ".png", bbox_inches="tight")
    plt.close(fig)
    paths += [str(p) + ".pdf", str(p) + ".png"]

    fig, ax = plt.subplots(figsize=(6.4, 3.6))
    years = sorted(df["year"].unique())
    for name, sub in df.groupby("paradigm"):
        ys = [(sub["year"] == y).sum() for y in years]
        ax.plot(years, ys, marker="o", label=name)
    ax.set_xlabel("Year")
    ax.set_ylabel("Papers")
    ax.set_title("Corpus coverage by year")
    ax.legend(frameon=False, fontsize=8)
    fig.tight_layout()
    p = folder / "survey_year"
    fig.savefig(str(p) + ".pdf", bbox_inches="tight")
    fig.savefig(str(p) + ".png", bbox_inches="tight")
    plt.close(fig)
    paths += [str(p) + ".pdf", str(p) + ".png"]

    fig, ax = plt.subplots(figsize=(6.8, 3.8))
    tab = pd.crosstab(df["data_topology"], df["control_loop"])
    im = ax.imshow(tab.values, cmap="Blues")
    ax.set_xticks(range(len(tab.columns)))
    ax.set_xticklabels(list(tab.columns), rotation=20, ha="right")
    ax.set_yticks(range(len(tab.index)))
    ax.set_yticklabels(list(tab.index))
    for i in range(tab.shape[0]):
        for j in range(tab.shape[1]):
            ax.text(j, i, int(tab.values[i, j]), ha="center", va="center", fontsize=9)
    ax.set_title("Taxonomy coverage (topology x control loop)")
    fig.colorbar(im, ax=ax, fraction=0.046)
    fig.tight_layout()
    p = folder / "survey_taxonomy"
    fig.savefig(str(p) + ".pdf", bbox_inches="tight")
    fig.savefig(str(p) + ".png", bbox_inches="tight")
    plt.close(fig)
    paths += [str(p) + ".pdf", str(p) + ".png"]

    labels = [r["label"] for r in RULES]
    # ordinal literature codes, not experimental accuracy
    quant = [5, 2, 4, 2]
    sem = [2, 5, 4, 5]
    rls = [5, 2, 3, 1]
    x = range(len(labels))
    w = 0.25
    fig, ax = plt.subplots(figsize=(7.0, 3.8))
    ax.bar([i - w for i in x], quant, w, label="Exact numbers", color="#0072B2")
    ax.bar(list(x), sem, w, label="Semantic / summary", color="#E69F00")
    ax.bar([i + w for i in x], rls, w, label="Row-level control", color="#009E73")
    ax.set_xticks(list(x))
    ax.set_xticklabels(labels, rotation=15, ha="right")
    ax.set_ylim(0, 6)
    ax.set_ylabel("Survey ordinal (1-5)")
    ax.set_title("When to use which (ordinal codes from the %d-paper reading)" % len(df))
    ax.legend(frameon=False, fontsize=8)
    fig.tight_layout()
    p = folder / "survey_decision"
    fig.savefig(str(p) + ".pdf", bbox_inches="tight")
    fig.savefig(str(p) + ".png", bbox_inches="tight")
    plt.close(fig)
    paths += [str(p) + ".pdf", str(p) + ".png"]

    fig, ax = plt.subplots(figsize=(6.4, 3.6))
    fam = df.apply(failure_family, axis=1).value_counts()
    ax.barh(list(fam.index)[::-1], list(fam.values)[::-1], color="#D55E00")
    ax.set_xlabel("Papers")
    ax.set_title("Dominant failure family (keyword coding, n=%d)" % len(df))
    fig.tight_layout()
    p = folder / "survey_failure"
    fig.savefig(str(p) + ".pdf", bbox_inches="tight")
    fig.savefig(str(p) + ".png", bbox_inches="tight")
    plt.close(fig)
    paths += [str(p) + ".pdf", str(p) + ".png"]

    fig, ax = plt.subplots(figsize=(6.2, 3.4))
    loops = df["control_loop"].value_counts()
    ax.bar(list(loops.index), list(loops.values), color="#0072B2")
    ax.set_ylabel("Papers")
    ax.set_title("Control loop in the coded set")
    fig.tight_layout()
    p = folder / "survey_loop"
    fig.savefig(str(p) + ".pdf", bbox_inches="tight")
    fig.savefig(str(p) + ".png", bbox_inches="tight")
    plt.close(fig)
    paths += [str(p) + ".pdf", str(p) + ".png"]
    return paths


def write_report(rows, folder="results"):
    folder = Path(folder)
    folder.mkdir(parents=True, exist_ok=True)
    df = frame(rows)
    csv_path = folder / "papers.csv"
    df.to_csv(csv_path, index=False)
    tex = latex_table(rows, folder / "table_survey.tex")
    mix = latex_mix(rows, folder / "table_mix.tex")
    tree = taxonomy(rows)
    ana = analysis(rows)
    (folder / "taxonomy.json").write_text(json.dumps(tree, indent=2), encoding="utf-8")
    (folder / "counts.json").write_text(json.dumps(counts(rows), indent=2), encoding="utf-8")
    (folder / "analysis.json").write_text(json.dumps(ana, indent=2), encoding="utf-8")
    paper_dir = ROOT / "paper" / "tex" / "tables"
    paper_dir.mkdir(parents=True, exist_ok=True)
    latex_mix(rows, paper_dir / "table_mix.tex")
    return str(csv_path), str(tex), str(mix)
