# Autonomous research and Harness storage

The agent chooses its own research approach, tool order and stopping point.
There are no prescribed stages, transitions, reproduction requirements, literature-review
intervals, batch summaries or continue/pivot decisions. No phase controls experiment submission
or report archival. Optional focus labels are free text for progress display only.

Harness owns literature text, notes, summaries, experiment code/checks/results and reports.
Use record_research_document, query_research_records and read_research_document as needed.
Use submit_experiment/poll_experiments for registered measurements and verify_experiment
for implementation checks. archive_research_report stores a report, while export_research_bundle
exports documents, code, raw outputs, plots, metrics CSV and reproduction information.
Evidence integrity, explicit user pause/direction commands, resource budgets and execution
checks remain in effect. Notes and summaries are available capabilities, not mandatory steps.

The former enforce_workflow setting and get_research_workflow tool have been removed.
Legacy YAML keys are ignored by the configuration loader. Existing phase labels and historical
stage records remain stored for reference but have no scheduling effect.

Storage: run_meta/system/research.sqlite3 and run_meta/system/artifacts/<sha256>.
GET /api/records supports kind/offset/limit for browsing history; the dashboard and companion
Q&A read the same store. Complete text is recoverable after context compaction or restart.

The existing python -m evo_core run command uses EvoCore. Upstream EvoScientist can use the
same tools via serve_harness.py and config.upstream-mcp.example.yaml; its real MCP client
is covered by subprocess integration tests. No global configuration is changed. Only one
owner runtime should use a run directory. Native upstream shell tools remain outside the
Harness tool boundary.

The long integration test demonstrates one possible research sequence, not an enforced
workflow. Regression coverage permits experiments without sources, summaries, decisions
or phase changes and verifies arbitrary focus labels and report delivery without stage history.
External PaperQA/Asta services, Temporal orchestration and automatic MLflow tracing are
separate integrations; Quarto rendering is optional for the exported report.qmd.
