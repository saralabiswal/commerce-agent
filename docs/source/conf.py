"""Sphinx configuration for CommerceAgent technical documentation."""

from __future__ import annotations

import sys
from pathlib import Path
from typing import Any

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))

project = "CommerceAgent Docs"
author = "Sarala Biswal"
release = "1.0.0"
html_title = "CommerceAgent Technical Documentation"
html_short_title = "CommerceAgent Docs"
html_show_sphinx = False
html_show_copyright = False

extensions = [
    "sphinx.ext.autodoc",
    "sphinx.ext.autosummary",
    "sphinx.ext.napoleon",
    "sphinx.ext.viewcode",
]

templates_path = ["_templates"]
exclude_patterns = []

html_theme = "furo"
html_static_path = ["_static"]
html_css_files = ["css/custom.css"]
html_theme_options = {
    "sidebar_hide_name": False,
    "light_css_variables": {
        "color-brand-primary": "#0f766e",
        "color-brand-content": "#0f766e",
        "color-api-name": "#0f172a",
        "color-api-pre-name": "#475569",
        "font-stack": "Inter, -apple-system, BlinkMacSystemFont, Segoe UI, sans-serif",
        "font-stack--monospace": "SFMono-Regular, Consolas, Liberation Mono, monospace",
    },
    "dark_css_variables": {
        "color-brand-primary": "#2dd4bf",
        "color-brand-content": "#5eead4",
        "color-api-name": "#f8fafc",
        "color-api-pre-name": "#cbd5e1",
    },
}

autodoc_default_options = {
    "members": True,
    "undoc-members": True,
    "show-inheritance": True,
    "exclude-members": "Field",
}
autodoc_typehints = "description"
autosummary_generate = True
napoleon_google_docstring = True
napoleon_numpy_docstring = True


def _format_signature(
    app: Any,
    what: str,
    name: str,
    obj: Any,
    options: Any,
    signature: str | None,
    return_annotation: str | None,
) -> tuple[str | None, str | None] | None:
    """Keep generated API pages readable by hiding noisy model constructors."""
    del app, name, obj, options

    if what == "class":
        return "()", return_annotation

    return signature, return_annotation


def _skip_internal_members(
    app: Any,
    what: str,
    name: str,
    obj: Any,
    skip: bool,
    options: Any,
) -> bool | None:
    """Hide internal/imported helpers that make the API reference noisy."""
    del app, what, obj, options

    if name in {"GraphState", "SettingsConfigDict"}:
        return True

    return skip


def setup(app: Any) -> None:
    app.connect("autodoc-skip-member", _skip_internal_members)
    app.connect("autodoc-process-signature", _format_signature)
