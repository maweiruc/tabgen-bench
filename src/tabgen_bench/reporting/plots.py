"""Agg/PNG plots of existing summaries. No raw rows or metric calculations."""

import base64
import os
import tempfile
from contextlib import contextmanager
from io import BytesIO


@contextmanager
def plotting_environment():
    """Keep font/config caches in a temporary writable directory, never the user's home."""
    with tempfile.TemporaryDirectory(prefix="tabbench-mpl-") as cache:
        previous = os.environ.get("MPLCONFIGDIR")
        os.environ["MPLCONFIGDIR"] = cache
        try:
            yield
        finally:
            if previous is None:
                os.environ.pop("MPLCONFIGDIR", None)
            else:
                os.environ["MPLCONFIGDIR"] = previous


def bars(labels, values, *, title, ylabel):
    # FigureCanvasAgg needs neither a GUI nor a writable global matplotlib font cache.
    from matplotlib.backends.backend_agg import FigureCanvasAgg
    from matplotlib.figure import Figure

    figure = Figure(figsize=(max(5, min(12, len(labels) * 0.65)), 3.4), layout="constrained")
    canvas = FigureCanvasAgg(figure)
    axes = figure.subplots()
    axes.bar(range(len(values)), values, color="#286a8c")
    axes.set_xticks(range(len(labels)), labels, rotation=30, ha="right")
    axes.set_title(title)
    axes.set_ylabel(ylabel)
    for label in (*axes.get_xticklabels(), axes.title, axes.yaxis.label):
        label.set_parse_math(False)
        label.set_usetex(False)
    buffer = BytesIO()
    canvas.print_png(buffer, metadata={"Software": "TabGenBench offline report"})
    return base64.b64encode(buffer.getvalue()).decode("ascii")
