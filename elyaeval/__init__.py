from .dataset import load_standard_dataset
from .fixtures import make_golden_context_fixture
from .report import evaluate_golden, new_report_path
from .html_report import metric_averages, read_csv_rows, render_html_report

__all__ = [
    "load_standard_dataset",
    "make_golden_context_fixture",
    "evaluate_golden",
    "new_report_path",
    "render_html_report",
    "metric_averages",
    "read_csv_rows",
]

# Note: metric presets (RAG_METRICS, etc.) are deliberately NOT re-exported
# here. Constructing a DeepEval metric object initializes the judge model
# immediately (raises if no model/API key is configured) — fine for a
# generated test file that's about to run evaluation, but not for `elyaeval
# init` or anything else that only needs the dataset loader. Import metric
# presets from elyaeval.metrics directly where they're actually used.