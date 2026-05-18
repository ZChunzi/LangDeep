"""Prometheus text-format exporter for LangDeep metrics snapshots."""

import re
from typing import Any, Dict, Iterable, List, Mapping, Tuple


class PrometheusMetricsExporter:
    """Convert ``MetricsCollector`` snapshots to Prometheus text format."""

    def __init__(self, namespace: str = "langdeep"):
        self._namespace = _sanitize_metric_name(namespace).strip("_")

    def export(self, snapshot: Mapping[str, Any]) -> str:
        """Return a Prometheus text exposition document."""
        lines: List[str] = []
        emitted_types = set()

        for raw_name, value in sorted(snapshot.get("counters", {}).items()):
            name, labels = _split_metric_key(raw_name)
            metric_name = self._metric_name(name)
            _emit_type(lines, emitted_types, metric_name, "counter")
            lines.append(f"{metric_name}{_format_labels(labels)} {_format_number(value)}")

        for raw_name, value in sorted(snapshot.get("gauges", {}).items()):
            name, labels = _split_metric_key(raw_name)
            metric_name = self._metric_name(name)
            _emit_type(lines, emitted_types, metric_name, "gauge")
            lines.append(f"{metric_name}{_format_labels(labels)} {_format_number(value)}")

        for raw_name, summary in sorted(snapshot.get("histograms", {}).items()):
            name, labels = _split_metric_key(raw_name)
            metric_name = self._metric_name(name)
            _emit_type(lines, emitted_types, metric_name, "summary")
            count = float(summary.get("count", 0))
            avg = float(summary.get("avg", 0))
            lines.append(f"{metric_name}_count{_format_labels(labels)} {_format_number(count)}")
            lines.append(f"{metric_name}_sum{_format_labels(labels)} {_format_number(avg * count)}")
            for quantile, field in (("0.5", "p50"), ("0.9", "p90"), ("0.99", "p99")):
                if field in summary:
                    q_labels = {**labels, "quantile": quantile}
                    lines.append(
                        f"{metric_name}{_format_labels(q_labels)} {_format_number(summary[field])}"
                    )
            for field in ("min", "max", "avg"):
                if field in summary:
                    gauge_name = f"{metric_name}_{field}"
                    _emit_type(lines, emitted_types, gauge_name, "gauge")
                    lines.append(
                        f"{gauge_name}{_format_labels(labels)} {_format_number(summary[field])}"
                    )

        return "\n".join(lines) + ("\n" if lines else "")

    def _metric_name(self, name: str) -> str:
        metric_name = _sanitize_metric_name(name)
        if not self._namespace:
            return metric_name
        return f"{self._namespace}_{metric_name}"


def export_prometheus_metrics(
    snapshot: Mapping[str, Any],
    *,
    namespace: str = "langdeep",
) -> str:
    """Convert a ``MetricsCollector.get_metrics()`` snapshot to Prometheus text."""
    return PrometheusMetricsExporter(namespace=namespace).export(snapshot)


def _split_metric_key(key: str) -> Tuple[str, Dict[str, str]]:
    if "|" not in key:
        return key, {}
    name, raw_labels = key.split("|", 1)
    labels: Dict[str, str] = {}
    for item in raw_labels.split(","):
        if not item or "=" not in item:
            continue
        label_name, value = item.split("=", 1)
        labels[_sanitize_label_name(label_name)] = value
    return name, labels


def _format_labels(labels: Mapping[str, str]) -> str:
    if not labels:
        return ""
    parts = [
        f'{_sanitize_label_name(key)}="{_escape_label_value(str(labels[key]))}"'
        for key in sorted(labels)
    ]
    return "{" + ",".join(parts) + "}"


def _emit_type(lines: List[str], emitted_types: set, metric_name: str, metric_type: str) -> None:
    marker = (metric_name, metric_type)
    if marker in emitted_types:
        return
    lines.append(f"# TYPE {metric_name} {metric_type}")
    emitted_types.add(marker)


def _sanitize_metric_name(name: str) -> str:
    sanitized = re.sub(r"[^a-zA-Z0-9_:]", "_", name)
    if not sanitized or not re.match(r"[a-zA-Z_:]", sanitized[0]):
        sanitized = f"_{sanitized}"
    return sanitized


def _sanitize_label_name(name: str) -> str:
    sanitized = re.sub(r"[^a-zA-Z0-9_]", "_", name)
    if not sanitized or not re.match(r"[a-zA-Z_]", sanitized[0]):
        sanitized = f"_{sanitized}"
    return sanitized


def _escape_label_value(value: str) -> str:
    return value.replace("\\", "\\\\").replace("\n", "\\n").replace('"', '\\"')


def _format_number(value: Any) -> str:
    number = float(value)
    if number.is_integer():
        return str(int(number))
    return repr(number)
