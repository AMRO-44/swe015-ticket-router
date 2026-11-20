"""Generate docs/architecture.png, results/results_chart.png and results/results_table.md
from the code structure and results/results.json."""
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
import matplotlib  # noqa: E402
matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
from matplotlib.patches import FancyBboxPatch  # noqa: E402

from src.config import DOCS_DIR, RESULTS_DIR, TEACHER_NAME, STUDENT_NAME  # noqa: E402
from src.results_io import load_results, MODEL_ORDER  # noqa: E402

INK, MUTED, BLUE, ORANGE, SURF = "#0b0b0b", "#52514e", "#2a78d6", "#eb6834", "#fcfcfb"
AREA = {"A": "#1baf7a", "B": "#2a78d6", "C": "#6250d6", "D": "#eb6834", "-": "#9a9893"}


def box(ax, x, y, w, h, title, body="", area="-", lw=1.4, bold=False):
    ax.add_patch(FancyBboxPatch((x, y), w, h, boxstyle="round,pad=0.008,rounding_size=0.012",
                                fc="white", ec=AREA[area], lw=lw * (1.8 if bold else 1)))
    ax.add_patch(FancyBboxPatch((x, y + h - 0.006), w, 0.006, boxstyle="square,pad=0", fc=AREA[area], ec="none"))
    ax.text(x + 0.008, y + h - 0.022, title, fontsize=8.6, weight="bold", color=INK, va="top")
    if body:
        ax.text(x + 0.008, y + h - 0.052, body, fontsize=7.1, color=MUTED, va="top", linespacing=1.35)


def arrow(ax, x1, y1, x2, y2, text=""):
    ax.annotate("", xy=(x2, y2), xytext=(x1, y1),
                arrowprops=dict(arrowstyle="-|>", color="#6b6a66", lw=1.1, shrinkA=1, shrinkB=1))
    if text:
        ax.text((x1 + x2) / 2, (y1 + y2) / 2 + 0.008, text, fontsize=6.5, color=MUTED, ha="center")


def architecture():
    fig = plt.figure(figsize=(13, 7.3), dpi=170)
    ax = fig.add_axes([0, 0, 1, 1])
    ax.set_xlim(0, 1); ax.set_ylim(0, 1); ax.axis("off")
    fig.patch.set_facecolor(SURF)
    ax.text(0.015, 0.975, "Architecture - Customer Support Ticket Router (inference path = what app.py runs)",
            fontsize=12, weight="bold", color=INK, va="top")
    ax.text(0.015, 0.94, "Runs on CPU (laptop or Colab VM) inside one Python process; Gradio serves the web UI. "
            "Border colour = technique area (A neural nets, B classical NLP, C Transformers, D efficient models).",
            fontsize=8, color=MUTED, va="top")

    # online path
    box(ax, 0.015, 0.70, 0.13, 0.15, "1. User / browser", "types a free-text\nmessage (EN / TR)\nin the Gradio UI")
    box(ax, 0.165, 0.66, 0.19, 0.19, "2. Preprocessing", "src/preprocess.py\n- regex normalization (TR chars,\n  URLs, repeated letters)\n"
        "- model-code regex  DW-6534\n- min. edit-distance correction\n  of product names (lexicon)", area="B")
    box(ax, 0.375, 0.73, 0.15, 0.12, "3. Small-talk rules", "src/eliza.py\nELIZA regex rules ->\nrule-based reply", area="B")

    ys = [0.555, 0.425, 0.295, 0.165]
    models = [("4a. TF-IDF + LogReg", "sklearn, word+char n-grams, FP64", "B", False),
              ("4b. word2vec + FFN", "own NumPy SGNS 100d -> 512 ReLU ->\nsoftmax, SGD+backprop, FP32", "A", False),
              (f"4c. {TEACHER_NAME.split('/')[-1]}", "fine-tuned, 12 layers, PyTorch FP32", "C", False),
              ("4d. Distilled BERT-small  INT8", f"{STUDENT_NAME.split('/')[-1]}, 4 layers\nKD from 4c + dynamic INT8 (Linear)", "D", True)]
    for (t, b, a, bold), y in zip(models, ys):
        box(ax, 0.375, y, 0.24, 0.105, t, b, area=a, bold=bold)
    ax.text(0.375, 0.135, "4d makes the routing decision (R5); 4a-4c run side by side for comparison.",
            fontsize=7, color=ORANGE, va="top", weight="bold")
    box(ax, 0.645, 0.36, 0.17, 0.24, "5. Post-processing", "src/router.py\n- softmax -> confidence\n- conf < tau (tuned on val)\n"
        "  -> out-of-scope\n- intent -> domain -> team\n- oos -> human agent\n- small_talk -> ELIZA reply")
    box(ax, 0.835, 0.36, 0.15, 0.24, "6. Output (Gradio)", "app.py\n- routed team + intent\n- confidence\n- bot reply\n"
        "- normalization info\n- 4-model table with\n  latency (ms)")

    arrow(ax, 0.145, 0.775, 0.165, 0.775)
    arrow(ax, 0.355, 0.79, 0.375, 0.79, "")
    for y in ys:
        arrow(ax, 0.30, 0.66, 0.375, y + 0.052)
    for y in ys:
        arrow(ax, 0.615, y + 0.052, 0.645, 0.48)
    arrow(ax, 0.525, 0.79, 0.70, 0.60, "greeting / small talk")
    arrow(ax, 0.815, 0.48, 0.835, 0.48)

    # offline path
    ax.plot([0.015, 0.985], [0.115, 0.115], color="#d8d7d2", lw=1, ls="--")
    ax.text(0.015, 0.105, "Offline training (train_all.py, Colab GPU)", fontsize=8.5, weight="bold", color=INK, va="top")
    steps = ["CLINC150 (150 intents + oos)\n+ 100 hand-written appliance msgs",
             "same preprocessing\ntrain / val / test split",
             "train 4a, 4b, 4c; distil 4c->student;\nquantize; tune tau on val",
             "evaluate all on SAME test set;\nsize + CPU latency (>=100 runs)",
             "models/ -> Hugging Face Hub\n-> download_models.py -> app"]
    for i, s in enumerate(steps):
        x = 0.015 + i * 0.197
        ax.text(x, 0.07, s, fontsize=7, color=MUTED, va="top", linespacing=1.3)
        if i:
            ax.annotate("", xy=(x - 0.006, 0.055), xytext=(x - 0.03, 0.055),
                        arrowprops=dict(arrowstyle="-|>", color="#6b6a66", lw=1))
    DOCS_DIR.mkdir(exist_ok=True)
    fig.savefig(DOCS_DIR / "architecture.png", facecolor=SURF)
    plt.close(fig)


def fmt(v, pct=False):
    if v is None:
        return "-"
    return f"{100 * v:.1f}%" if pct else f"{v:g}"


def results_table():
    r = load_results()
    ms = r.get("models", {})
    lines = [f"Hardware for latency: {r.get('hardware', 'n/a')}", "",
             "| Model | Role | Precision | In-scope acc. | OOS recall | Overall acc. | Size (MB) | Latency (ms, mean of N runs) |",
             "|---|---|---|---|---|---|---|---|"]
    for mid in MODEL_ORDER:
        m = ms.get(mid)
        if not m:
            continue
        lines.append(f"| {m['name']} | {m['role']} | {m['precision']} | {fmt(m['in_scope_acc'], True)} | "
                     f"{fmt(m['oos_recall'], True)} | {fmt(m['overall_acc'], True)} | {m['size_mb']} | "
                     f"{m['latency_ms']:.2f} (N={m['latency_runs']}) |")
    RESULTS_DIR.mkdir(exist_ok=True)
    (RESULTS_DIR / "results_table.md").write_text("\n".join(lines) + "\n", encoding="utf-8")
    return ms


def results_chart(ms):
    ids = [m for m in MODEL_ORDER if m in ms]
    if not ids:
        return
    names = [ms[m]["name"] for m in ids][::-1]
    colors = [ORANGE if m == "student_int8" else BLUE for m in ids][::-1]
    panels = [("Overall accuracy (%)", [100 * ms[m]["overall_acc"] for m in ids][::-1], "{:.1f}"),
              ("Model size (MB)", [ms[m]["size_mb"] for m in ids][::-1], "{:.1f}"),
              ("CPU latency per message (ms)", [ms[m]["latency_ms"] for m in ids][::-1], "{:.2f}")]
    fig, axes = plt.subplots(1, 3, figsize=(13, 0.55 * len(ids) + 1.3), dpi=170, sharey=True)
    fig.patch.set_facecolor(SURF)
    for ax, (title, vals, f) in zip(axes, panels):
        ax.barh(names, vals, color=colors, height=0.55, edgecolor=SURF, linewidth=2)
        ax.set_title(title, fontsize=10, color=INK, loc="left", weight="bold")
        for i, v in enumerate(vals):
            ax.text(v, i, " " + f.format(v), va="center", fontsize=8, color=MUTED)
        ax.set_xlim(0, max(vals) * 1.25)
        for s in ["top", "right"]:
            ax.spines[s].set_visible(False)
        ax.spines["left"].set_color("#bdbcb7"); ax.spines["bottom"].set_color("#bdbcb7")
        ax.tick_params(colors=MUTED, labelsize=8)
        ax.set_facecolor(SURF)
        ax.xaxis.grid(True, color="#ecebe7", lw=0.8); ax.set_axisbelow(True)
    fig.text(0.01, 0.005, "Orange = model used by the app (distilled + INT8). Same held-out test set for all models.",
             fontsize=8, color=MUTED)
    fig.tight_layout(rect=(0, 0.04, 1, 1))
    fig.savefig(RESULTS_DIR / "results_chart.png", facecolor=SURF)
    plt.close(fig)


if __name__ == "__main__":
    architecture()
    results_chart(results_table())
    print("figures written to docs/ and results/")
