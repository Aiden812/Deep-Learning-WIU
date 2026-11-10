"""Run qwen_clean_and_label.ipynb from a terminal instead of a Jupyter kernel.

The notebook's code cells are executed in order in this Python process, so a long run does not depend on VS Code
keeping its kernel connection alive. The image-grid cells are skipped. All answers go to the same caches as the
notebook, so you can switch between the two freely; both resume where the last run stopped.

  python models/classifier/scripts/Daniel/run_qwen_pipeline.py                  # run everything that is left
  python models/classifier/scripts/Daniel/run_qwen_pipeline.py --concurrency 3  # more requests at once
  python models/classifier/scripts/Daniel/run_qwen_pipeline.py --limit 200      # at most 200 images this run
  python models/classifier/scripts/Daniel/run_qwen_pipeline.py --outputs-only   # send nothing, just write outputs

Ctrl+C once stops sending (requests already in flight finish and are saved) and still writes the outputs
(external_clean, labels.csv, YOLO folder, the filtered, split yolo_filtered/ + labels_filtered.csv, and
oral_dataset.zip).
Ctrl+C again quits straight away.
While it runs, Windows is kept from going to sleep.
"""
import argparse
import ctypes
import json
import os
import sys
from pathlib import Path

REPO = Path(__file__).resolve().parents[4]
NOTEBOOK = REPO / "models" / "classifier" / "notebooks" / "Daniel" / "qwen_clean_and_label.ipynb"
SKIP_PREFIXES = ("# Visual check", "# Show some near-duplicate groups")   # image grids: no use in a terminal
RUN_CELL_MARKER = "run_queue(todo, process_image)"


def keep_awake(on):
    if sys.platform == "win32":
        ES_CONTINUOUS, ES_SYSTEM_REQUIRED = 0x80000000, 0x00000001
        ctypes.windll.kernel32.SetThreadExecutionState(ES_CONTINUOUS | (ES_SYSTEM_REQUIRED if on else 0))


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--notebook", type=Path, default=NOTEBOOK)
    ap.add_argument("--concurrency", type=int, help="requests in flight at once (notebook default: 2)")
    ap.add_argument("--limit", type=int, help="max images to process this run")
    ap.add_argument("--no-screen", action="store_true", help="don't send screen requests")
    ap.add_argument("--no-label", action="store_true", help="don't send label requests")
    ap.add_argument("--no-boxes", action="store_true", help="don't send box requests (label and known-condition routes)")
    ap.add_argument("--no-check", action="store_true", help="don't send the light check (dentalai / zenodo)")
    ap.add_argument("--box-eval", type=int, metavar="N", help="also run the Box check on N human-boxed images")
    ap.add_argument("--outputs-only", action="store_true", help="send nothing; rebuild the outputs from saved answers")
    ap.add_argument("--no-zip", action="store_true", help="don't build oral_dataset.zip at the end")
    args = ap.parse_args()
    if args.no_zip:
        os.environ["QWEN_MAKE_ZIP"] = "0"
    if args.box_eval is not None:
        os.environ["QWEN_BOX_EVAL_N"] = str(args.box_eval)

    # The notebook reads its settings from QWEN_* environment variables (see its parameters cell).
    os.environ.setdefault("DLWIU_ROOT", str(REPO))
    if args.concurrency is not None:
        os.environ["QWEN_CONCURRENCY"] = str(args.concurrency)
    if args.limit is not None:
        os.environ["QWEN_LIMIT"] = str(args.limit)
    for step, off in [("SCREEN", args.no_screen), ("LABEL", args.no_label), ("BOXES", args.no_boxes),
                      ("CHECK", args.no_check)]:
        if off or args.outputs_only:
            os.environ[f"QWEN_RUN_{step}"] = "0"

    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    plt.show = lambda *a, **k: plt.close("all")

    nb = json.loads(args.notebook.read_text(encoding="utf-8"))
    ns = {"__name__": "__main__", "display": print}
    keep_awake(True)
    try:
        for i, cell in enumerate(nb["cells"]):
            if cell["cell_type"] != "code":
                continue
            src = "".join(cell["source"]) if isinstance(cell["source"], list) else cell["source"]
            if src.lstrip().startswith(SKIP_PREFIXES):
                continue
            src = src.replace("from IPython.display import display\n", "")
            try:
                exec(compile(src, f"{args.notebook.name} [cell {i}]", "exec"), ns)
            except KeyboardInterrupt:
                if RUN_CELL_MARKER not in src:
                    raise
                print("\nStopped sending. Writing outputs from the answers saved so far "
                      "(Ctrl+C again to quit now) ...", flush=True)
    except KeyboardInterrupt:
        print("\nQuit.")
        return 130
    finally:
        keep_awake(False)
    return 0


if __name__ == "__main__":
    sys.exit(main())
