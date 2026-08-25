"""
Auto-registered pytest plugin (see the [project.entry-points.pytest11]
entry in pyproject.toml) — this is what makes HTML reports + per-metric
averages appear with NO changes to generated test files or CI YAML beyond
bumping the installed elyaeval version. Nothing in here needs to be
imported explicitly anywhere; pytest discovers and loads it automatically
for any environment that has elyaeval installed.

At the end of the pytest session, for every CSV that
report.append_csv_rows / tracing_report.append_traced_csv_rows wrote to
during THIS run (tracked via html_report.register_report), this:
  1. renders an HTML sibling next to that CSV (same name, .html extension)
  2. prints a per-metric (or per-span+metric, for component CSVs) average
     table to the terminal, right after pytest's own summary

Only touches CSVs that were actually written this session — a stale
results_*.csv left over from a previous run in the same report/ directory
is not re-rendered or double-counted.
"""

from __future__ import annotations

from .html_report import print_metric_averages, read_csv_rows, render_html_report, session_reports


def pytest_sessionfinish(session, exitstatus) -> None:
    reports = session_reports()
    if not reports:
        return

    terminalreporter = session.config.pluginmanager.get_plugin("terminalreporter")

    def _write(line: str = "") -> None:
        if terminalreporter is not None:
            terminalreporter.write_line(line)
        else:
            print(line)

    _write("")
    _write("=" * 20 + " elyaeval report " + "=" * 20)
    for csv_path, group_keys in reports.items():
        if not csv_path.exists():
            continue
        try:
            html_path = render_html_report(csv_path, group_keys=group_keys)
        except Exception as exc:  # noqa: BLE001 — reporting must never fail the run
            _write(f"[elyaeval] could not render HTML for {csv_path}: {exc}")
            continue

        _write(f"[elyaeval] {csv_path} -> {html_path.name}")
        rows = read_csv_rows(csv_path)
        print_metric_averages(rows, group_keys=group_keys, write=_write)
    _write("")
