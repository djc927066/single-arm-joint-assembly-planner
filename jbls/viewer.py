"""Write the offline mesh viewer for scene and discrete assembly states.

The viewer deliberately does not interpolate robot or part motion.  Its HTML,
JavaScript, shaders and mesh data are embedded in one portable output file.
"""
import json
from pathlib import Path


def write_viewer(report, path):
    """Embed a JSON report into the adjacent, dependency-free HTML template."""
    template = Path(__file__).with_name("scene_viewer.html").read_text(encoding="utf-8")
    # Escaping '<' prevents report text from closing the application/json script.
    data = json.dumps(report, ensure_ascii=False, allow_nan=False).replace("<", "\\u003c")
    marker = "__JBLS_REPORT_JSON__"
    if template.count(marker) != 1:
        raise ValueError("scene_viewer.html must contain exactly one report placeholder")
    Path(path).write_text(template.replace(marker, data), encoding="utf-8")
