import argparse
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))
import survey

ROOT = Path(__file__).resolve().parent.parent


def parse_args():
    p = argparse.ArgumentParser()
    sub = p.add_subparsers(dest="cmd", required=True)

    s = sub.add_parser("survey", help="synthesize the 80-paper corpus")
    s.add_argument("--papers", default=None)
    s.add_argument("--results", default=str(ROOT / "results"))
    s.add_argument("--figs", default=str(ROOT / "figs"))

    m = sub.add_parser("merge", help="append data/extra_papers.json into data/papers.json")
    m.add_argument("--core", default=None)
    m.add_argument("--extra", default=None)
    m.add_argument("--dest", default=None)

    d = sub.add_parser("decide", help="when to use SQL vs RAG")
    d.add_argument("--q", default="", help="business question")
    d.add_argument("--data", default="auto", choices=["auto", "tabular", "text", "both"])
    d.add_argument("--has-schema", action="store_true")
    d.add_argument("--has-docs", action="store_true")
    d.add_argument("--rls", action="store_true")
    d.add_argument("--papers", default=None)
    return p.parse_args()


def main():
    args = parse_args()
    if args.cmd == "merge":
        rows = survey.merge_corpus(args.core, args.extra, args.dest)
        print("merged", len(rows))
        return
    rows = survey.load_papers(args.papers)
    if args.cmd == "survey":
        print("papers", len(rows))
        print(survey.counts(rows))
        survey.print_taxonomy(survey.taxonomy(rows))
        print(survey.write_report(rows, args.results))
        fig_dir = args.figs
        for p in survey.draw(rows, fig_dir):
            print(p)
        paper_figs = ROOT / "paper" / "figs"
        paper_figs.mkdir(parents=True, exist_ok=True)
        import shutil

        for name in (
            "survey_paradigm",
            "survey_year",
            "survey_taxonomy",
            "survey_decision",
            "survey_failure",
            "survey_loop",
        ):
            for ext in (".pdf", ".png"):
                src = Path(fig_dir) / (name + ext)
                if src.exists():
                    shutil.copy2(src, paper_figs / src.name)
        return
    d = survey.decide(
        rows,
        q=args.q,
        data=args.data,
        has_schema=args.has_schema,
        has_docs=args.has_docs,
        rls=args.rls,
    )
    survey.print_decision(d)


if __name__ == "__main__":
    main()
