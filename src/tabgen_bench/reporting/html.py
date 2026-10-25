"""Offline, escaped rendering of normalized records and persisted summaries only."""

import html
import json
from collections import Counter

from .plots import bars

TEMPLATE_VERSION = "offline_html_v1.0.0"
DOMAINS = ("validity", "fidelity", "diversity", "utility", "privacy", "statistical")


def escape(value):
    return html.escape(str(value), quote=True)


def display(value):
    return "NA" if value is None else str(value)


def table(headers, rows):
    return (
        "<div class='scroll'><table><thead><tr>"
        + "".join(f"<th>{escape(h)}</th>" for h in headers)
        + "</tr></thead><tbody>"
        + "".join(
            "<tr>" + "".join(f"<td>{escape(display(c))}</td>" for c in row) + "</tr>"
            for row in rows
        )
        + "</tbody></table></div>"
    )


def figure(labels, values, title, ylabel):
    if not values:
        return "<p>No numeric values recorded for this plot.</p>"
    payload = bars(labels, values, title=title, ylabel=ylabel)
    return f"<figure><img alt='{escape(title)}' src='data:image/png;base64,{payload}'></figure>"


def _dataset(record):
    m, meta = record.manifest, record.metadata
    schema = meta["dataset"]["dataset_schema"]
    columns = schema["columns"]
    output = "<h2>Dataset / frozen split</h2>" + table(
        ("Dataset", "Version", "Source", "Rows", "Synthetic rows", "Split seed", "Split counts"),
        [
            (
                m.dataset_id,
                m.dataset_version,
                f"{m.source.provider}:{m.source.source_id} v{m.source.version}",
                meta["dataset"].get("expected_rows"),
                m.n_synthetic,
                m.seeds.split_seed,
                json.dumps(meta["split_counts"], ensure_ascii=False, sort_keys=True),
            )
        ],
    )
    output += (
        "<p>One frozen real split; replicate seeds do not represent independent real splits.</p>"
    )
    output += table(
        ("Column", "Role", "Logical type", "Nullable", "Rule source", "Rule version"),
        [
            (
                c["name"],
                c["role"],
                c["logical_type"],
                c["nullable"],
                c["constraint_source"],
                c["rule_version"],
            )
            for c in columns
        ],
    )
    overlap = meta.get("provenance", {}).get("row_overlap", {})
    if meta.get("provenance", {}).get("identical_real_file_contents"):
        output += "<p class='warning'>Identical real file contents across declared splits; "
        output += "independent holdout calibration is not established.</p>"
    if overlap:
        output += "<p class='warning'>Known source duplicate / cross-split record metadata: "
        output += escape(json.dumps(overlap, ensure_ascii=False, sort_keys=True)) + "</p>"
        output += "<p>Frozen row positions retained. No regrouping or source deduplication.</p>"
    summaries = record.report_input.get("table_summaries", {})
    if summaries:
        output += "<h3>Recorded missingness / target distribution</h3>" + table(
            ("Input", "Rows", "Missing counts", "Target counts"),
            [
                (
                    name,
                    data["n_rows"],
                    json.dumps(data["missing_counts"], ensure_ascii=False),
                    json.dumps(data.get("target_counts", {}), ensure_ascii=False),
                )
                for name, data in summaries.items()
            ],
        )
        for name, data in summaries.items():
            if data.get("warning"):
                output += f"<p class='warning'>{escape(name)}: {escape(data['warning'])}</p>"
        # Histograms were prepared by the execution service, before reporting.
        for name, data in summaries.items():
            for column, counts in data.get("distributions", {}).items():
                output += figure(
                    list(counts), list(counts.values()), f"{name}: {column}", "Recorded count"
                )
    else:
        lock = meta.get("provenance", {}).get("source_lock", {})
        output += "<p>Raw distributions were not recorded in this legacy report input; "
        output += "report redraw does not read data tables or reconstruct them.</p>"
        for key in ("target_counts", "missing_counts"):
            if key in lock:
                output += table(
                    ("Source summary", "Recorded value"),
                    [(key, json.dumps(lock[key], ensure_ascii=False, sort_keys=True))],
                )
    return output


def _validity(record):
    output = "<h3>Validity reason breakdown</h3>"
    report = next(
        (m.details.to_dict() for m in record.metrics if m.metric_id == "validity.schema_pass"), {}
    )
    output += table(
        ("Invalid cells", "Checked cells (including target)", "Invalid rows", "Rows"),
        [
            (
                report.get("invalid_cell_count"),
                report.get("checked_cell_count"),
                report.get("invalid_row_count"),
                report.get("n_rows"),
            )
        ],
    )
    reasons = report.get("column_reasons", [])
    output += table(
        ("Column", "Reason code", "Count", "Rate", "Rule version", "Constraint source"),
        [
            (
                r["column"],
                r["reason_code"],
                r["count"],
                r["rate"],
                r["rule_version"],
                r["constraint_source"],
            )
            for r in reasons
        ],
    )
    structural = report.get("structural_errors", [])
    output += table(
        ("Structural reason", "Recorded details"),
        [
            (r.get("reason_code"), json.dumps(r, ensure_ascii=False, sort_keys=True))
            for r in structural
        ],
    )
    if report.get("schema_pass") == 0:
        output += "<p class='warning'>Schema failed; dependent metrics may be "
        output += "not_applicable: blocked_by_validity. No implicit repair.</p>"
    return output


def render(records, aggregation):
    pieces = [
        """<!doctype html><html lang="en"><head><meta charset="utf-8">
<meta name="viewport" content="width=device-width, initial-scale=1">
<meta http-equiv="Content-Security-Policy"
content="default-src 'none'; img-src data:; style-src 'unsafe-inline'">
<title>TabGenBench offline report</title><style>
body{font:15px system-ui,sans-serif;color:#173042;background:#f5f8fa;
margin:0 auto;padding:32px;max-width:1400px}
h1,h2,h3{color:#143e58}section{background:white;padding:20px;margin:20px 0;border-radius:8px}
table{border-collapse:collapse;width:100%;font-size:13px}
td,th{padding:8px;text-align:left;border-bottom:1px solid #dce5eb;
vertical-align:top;overflow-wrap:anywhere}
th{background:#e9f1f6}.scroll{overflow:auto}img{max-width:100%;height:auto}figure{margin:20px 0}
.warning{background:#fff0cc;border-left:4px solid #bf6e0b;padding:12px}
code{overflow-wrap:anywhere}p{line-height:1.6}
</style></head><body><h1>TabGenBench · Offline evaluation report</h1>"""
    ]
    pieces.append(f"<p>Template {escape(TEMPLATE_VERSION)}. Persisted results only.</p>")
    pieces.append(
        "<p class='warning'>DCR / Exact Copy / Train-Closer are proximity diagnostics, "
        "not a privacy guarantee or a privacy attack success rate. "
        "DCR 是接近度诊断，不代表隐私保证。</p>"
    )
    pieces.append(
        "<section><h2>Randomness audit / comparison</h2><p>All summaries are descriptive. "
        "Independence is not established; no Monte Carlo standard errors or confidence "
        "intervals are computed from the number of seeds. Unknown is not zero.</p>"
    )
    for group in aggregation["groups"]:
        generator = group["generator"]
        pieces.append(f"<h3>{escape(generator['id'])}</h3>")
        counts = (
            "n_planned",
            "n_attempted",
            "n_completed",
            "n_unique_outputs",
            "n_independent_replicates",
        )
        pieces.append(
            table(
                counts,
                [[group[n] if group[n] is not None else "unconfirmed (null)" for n in counts]],
            )
        )
        pieces.append(f"<p>{escape(group['planning_scope'])}; real_split_replicates=1.</p>")
        if (
            group["n_unique_outputs"] is not None
            and group["n_unique_outputs"] < group["n_completed"]
        ):
            pieces.append(
                "<p class='warning'>Repeated synthetic outputs detected: unique output "
                "count is smaller than completed run count. Seeds / replayed runs "
                "must not be counted as independent Monte Carlo replicates.</p>"
            )
        if group["n_outputs_without_multiset_audit"]:
            pieces.append(
                "<p class='warning'>Output uniqueness audit unavailable for illegal "
                "scalars; unique output count is unknown, not zero.</p>"
            )
        pieces.append(
            table(
                ("Attempt status counts", "Latest cell status counts"),
                [
                    (
                        json.dumps(group["attempt_status_counts"], sort_keys=True),
                        json.dumps(group["cell_status_counts"], sort_keys=True),
                    )
                ],
            )
        )
        pieces.append(
            table(
                ("Output hash", "Run IDs", "Replicate seeds"),
                [
                    (
                        d["synthetic_multiset_sha256"],
                        ", ".join(d["run_ids"]),
                        str(d["replicate_seeds"]),
                    )
                    for d in group["duplicate_outputs"]
                ],
            )
        )
    pieces.append("<h3>Metric comparison (separate scales; no composite score)</h3>")
    pieces.append(
        table(
            (
                "Generator",
                "Metric",
                "Domain",
                "Direction",
                "Unit",
                "Mean",
                "Descriptive SD",
                "Unique units",
                "Statuses (all attempts)",
            ),
            [
                (
                    g["generator"]["id"],
                    m["metric_id"],
                    m["metric_domain"],
                    m["direction"],
                    m["unit"],
                    m["mean"],
                    m["std"],
                    m["n_unique_output_units"],
                    json.dumps(m["status_counts"], sort_keys=True),
                )
                for g in aggregation["groups"]
                for m in g["metrics"]
            ],
        )
    )
    ids = list(dict.fromkeys(m["metric_id"] for g in aggregation["groups"] for m in g["metrics"]))
    for metric_id in ids:
        items = [
            (g["generator"]["id"], m)
            for g in aggregation["groups"]
            for m in g["metrics"]
            if m["metric_id"] == metric_id and m["mean"] is not None
        ]
        if items:
            pieces.append(
                figure(
                    [name for name, _ in items],
                    [m["mean"] for _, m in items],
                    metric_id,
                    f"{items[0][1]['unit']} · {items[0][1]['direction']}",
                )
            )
    pieces.append("</section>")
    for record in records:
        m = record.manifest
        pieces.append(f"<section><h2>Run {escape(m.run_id)}</h2>")
        pieces.append(
            table(
                ("Status", "Generator", "Version", "Parameters", "Failure"),
                [
                    (
                        m.status,
                        m.generator_id or record.metadata.get("synthetic_label", "external"),
                        m.generator_version,
                        json.dumps(m.generator_params.to_dict(), ensure_ascii=False),
                        f"{m.error_type or ''} {m.error_message or ''}",
                    )
                ],
            )
        )
        pieces.append(_dataset(record))
        if record.audit:
            environment = record.audit.get("environment", {})
            audit_rows = [
                (key, record.audit.get(key))
                for key in (
                    "synthetic_content_sha256",
                    "synthetic_multiset_sha256",
                    "monte_carlo_independence",
                    "sample_seed_independence_assumed",
                    "reference_sampling_seed_semantics",
                    "seed_note",
                    "notes",
                )
                if key in record.audit
            ]
            audit_rows += [
                (key, environment[key])
                for key in (
                    "fit_seed",
                    "sample_requested_seed",
                    "sdk_sampling_seed",
                    "sample_seed_semantics",
                    "train_rows",
                    "parameters",
                )
                if key in environment
            ]
            pieces.append("<h3>Persisted synthetic output / SDK audit</h3>")
            pieces.append(table(("Audit field", "Recorded value"), audit_rows))
        pieces.append("<h3>Training / sampling / evaluation configuration</h3>")
        pieces.append(
            table(
                ("Seed / configuration", "Recorded value"),
                [
                    *m.seeds.model_dump().items(),
                    ("n_synthetic", m.n_synthetic),
                    ("suite", f"{m.suite_id} v{m.suite_version}"),
                    ("device", m.environment.actual_device),
                    ("reproducibility", m.environment.reproducibility_level),
                ],
            )
        )
        pieces.append(
            table(
                ("Metric status", "Count"),
                sorted(Counter(r.status for r in record.metrics).items()),
            )
        )
        for domain in DOMAINS:
            metrics = [r for r in record.metrics if r.metric_domain == domain]
            pieces.append(f"<h3>{escape(domain.title())}</h3>")
            if not metrics:
                expected = [
                    item["id"]
                    for item in record.metadata["evaluation"]["metrics"]
                    if item["domain"] == domain
                ]
                if expected:
                    pieces.append(
                        "<p>Included in this suite; no persisted MetricResult. "
                        "Run failed or is incomplete; see run status.</p>"
                    )
                    pieces.append(table(("Metric", "Status"), [(m, "missing") for m in expected]))
                else:
                    pieces.append(
                        "<p>Not included in this suite（未纳入当前 Suite）; no score.</p>"
                    )
                continue
            pieces.append(
                "<div class='scroll'><table><thead><tr>"
                + "".join(
                    f"<th>{escape(s)}</th>"
                    for s in (
                        "Metric",
                        "Value",
                        "Status",
                        "Direction",
                        "Unit",
                        "Reference",
                        "Version",
                        "Notes / error",
                    )
                )
                + "</tr></thead><tbody>"
            )
            for r in metrics:
                pieces.append(
                    f"<tr data-run='{escape(m.run_id)}' data-metric='{escape(r.metric_id)}' "
                    f"data-value='{escape('' if r.value is None else repr(r.value))}'>"
                )
                values = (
                    r.metric_id,
                    display(r.value),
                    r.status,
                    r.direction,
                    r.unit,
                    (
                        r.reference_split
                        + (
                            ": "
                            + ", ".join(f"{i.input_name}/{i.reference_split}" for i in r.references)
                            if r.references
                            else ""
                        )
                    ),
                    r.metric_version,
                    " / ".join(s for s in (r.notes, r.error_type, r.error_message) if s),
                )
                pieces.append("".join(f"<td>{escape(v)}</td>" for v in values) + "</tr>")
            pieces.append("</tbody></table></div>")
        pieces.append(
            "<p>Formal Privacy Attacks: not included in the built-in V1 suite; no score.</p>"
        )
        pieces.append(_validity(record))
        pieces.append("<h3>Recorded per-column marginal diagnostics</h3>")
        for r in record.metrics:
            values = r.details.to_dict().get("per_column", {})
            if values:
                pieces.append(
                    table(
                        ("Metric", "Column", "Value"),
                        [(r.metric_id, col, val) for col, val in values.items()],
                    )
                )
                pieces.append(figure(list(values), list(values.values()), r.metric_id, r.unit))
        pieces.append(
            "<p>C2ST reports held-out discriminator AUC (target ≈ 0.5); "
            "Real-vs-Real calibration uses validation/test when available. "
            "TRTR and TSTR share real test; Performance Gap is signed TSTR − TRTR.</p>"
        )
        pieces.append("<h3>Artifact identity / cache / experiment versions</h3>")
        pieces.append(
            table(
                ("Identity", "Value"),
                [
                    ("manifest_sha256", record.manifest_sha256),
                    ("config_hash", m.config_hash),
                    *m.hashes.model_dump().items(),
                    ("git_commit", m.git_commit),
                    ("git_dirty", m.git_dirty),
                    ("Python", m.environment.python),
                    ("OS", m.environment.operating_system),
                ],
            )
        )
        pieces.append(
            table(
                ("Artifact", "SHA256", "Bytes"),
                [(a.path, a.sha256, a.size_bytes) for a in m.artifacts],
            )
        )
        pieces.append(
            table(("Cache stage", "Key", "Hit"), [(c.stage, c.key, c.hit) for c in m.cache])
        )
        pieces.append(
            table(("Library", "Version"), [(v.name, v.version) for v in m.environment.libraries])
        )
        pieces.append("</section>")
    pieces.append("</body></html>")
    return "\n".join(pieces)
