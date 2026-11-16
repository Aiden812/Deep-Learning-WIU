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
# Code cells whose first line starts with one of these are visual-only (image grids with plt.show); in a terminal
# they cost a full image decode per tile and display nothing useful, so they are skipped entirely.
SKIP_PREFIXES = ("# Visual check", "# Show some near-duplicate groups")
# The long "run the queue" cell catches KeyboardInterrupt itself (it drains the worker threads and lets the output
# cells still run). The runner recognises that cell by this statement and hands the Ctrl+C over to it instead of
# aborting; if the notebook is edited and the call changes shape, this marker must be updated to match.
RUN_CELL_MARKER = "run_queue(todo, process_image)"


def keep_awake(on):
    """Tell Windows to stay awake (or allow sleep again) while the pipeline runs.

    Args:
        on: True to request "system must not sleep", False to release the request.

    Uses SetThreadExecutionState with ES_CONTINUOUS, which suspends the machine's idle-sleep timer for this
    process without touching the user's power plan; the flag clears itself when the process exits. On every
    non-Windows platform this is a no-op (macOS/Linux laptops are expected to be plugged in anyway).
    """
    if sys.platform == "win32":
        ES_CONTINUOUS, ES_SYSTEM_REQUIRED = 0x80000000, 0x00000001
        ctypes.windll.kernel32.SetThreadExecutionState(ES_CONTINUOUS | (ES_SYSTEM_REQUIRED if on else 0))


def main():
    """Parse CLI options into QWEN_* env vars, then exec every code cell of the notebook in order.

    Flow:
      1. CLI flags -> QWEN_* environment variables (the notebook's parameters cell reads them; env is the
         single source of truth so terminal and Jupyter runs cannot disagree about their settings).
      2. Force headless matplotlib so the notebook's plotting cells neither open windows nor block.
      3. exec each code cell into one shared namespace, exactly like a Jupyter kernel: cell 2's imports and
         find_repo_root() are visible to every later cell, and caches/resume behave as in the notebook.
      4. On Ctrl+C: inside the queue cell the notebook handles it gracefully (see RUN_CELL_MARKER), anywhere
         else the run aborts.

    Returns:
        int: process exit code - 0 when the notebook ran to completion (or Ctrl+C still wrote the outputs),
        130 when a second Ctrl+C quit the run before the outputs were written.
    """
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
    # Two switches the notebook also honours from the environment (not step toggles, so set directly):
    if args.no_zip:
        os.environ["QWEN_MAKE_ZIP"] = "0"
    if args.box_eval is not None:
        os.environ["QWEN_BOX_EVAL_N"] = str(args.box_eval)

    # The notebook reads its settings from QWEN_* environment variables (see its parameters cell).
    # DLWIU_ROOT only gets a default: an explicit value from the shell (or a non-default --notebook sitting
    # outside the repo) must keep working, so setdefault rather than overwrite.
    os.environ.setdefault("DLWIU_ROOT", str(REPO))
    if args.concurrency is not None:
        os.environ["QWEN_CONCURRENCY"] = str(args.concurrency)
    if args.limit is not None:
        os.environ["QWEN_LIMIT"] = str(args.limit)
    # Step toggles: --outputs-only is "all steps off", so it implies every QWEN_RUN_* = 0 below.
    for step, off in [("SCREEN", args.no_screen), ("LABEL", args.no_label), ("BOXES", args.no_boxes),
                      ("CHECK", args.no_check)]:
        if off or args.outputs_only:
            os.environ[f"QWEN_RUN_{step}"] = "0"

    # Headless matplotlib: "Agg" renders to memory/file instead of a GUI backend, and plt.show() is replaced by
    # close("all") so the notebook's grid cells neither open windows nor wait for one to be closed.
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    plt.show = lambda *a, **k: plt.close("all")

    # One shared namespace for all cells, like a Jupyter kernel: later cells see names defined by earlier ones.
    # display() is not available outside IPython, so it is aliased to print in the cell namespace.
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
            # The notebook imports IPython's display for its markdown tables; outside a kernel the import itself
            # would fail, so it is stripped (display was already aliased to print above).
            src = src.replace("from IPython.display import display\n", "")
            try:
                # compile() with the notebook cell as filename: tracebacks point at "<notebook> [cell N]" instead
                # of a bare exec, which makes failures reportable without opening Jupyter.
                exec(compile(src, f"{args.notebook.name} [cell {i}]", "exec"), ns)
            except KeyboardInterrupt:
                # Ctrl+C during the long work cell: hand it to the notebook itself (its run_queue catches it,
                # stops feeding new work, lets in-flight requests finish and saves them); everywhere else, a
                # Ctrl+C means "abort" - propagate to the outer handler and exit non-zero.
                if RUN_CELL_MARKER not in src:
                    raise
                print("\nStopped sending. Writing outputs from the answers saved so far "
                      "(Ctrl+C again to quit now) ...", flush=True)
    except KeyboardInterrupt:
        # Second Ctrl+C (or one outside the queue cell): stop immediately; unsaved answers are lost, but all
        # step caches written up to now are intact, so the next run resumes from them.
        print("\nQuit.")
        return 130
    finally:
        keep_awake(False)
    return 0


if __name__ == "__main__":
    sys.exit(main())
