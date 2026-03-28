"""
Marp CLI wrapper for exporting slide decks to PDF and HTML.

Requires Node.js and ``@marp-team/marp-cli`` available via npx.
Falls back gracefully if Marp CLI is not installed.

Usage::

    from src.presentation.marp_export import export_pdf, check_ready

    status = check_ready()
    if status["marp_cli"]:
        pdf_path = export_pdf("outputs/deck.marp.md")
"""

from __future__ import annotations

import subprocess
from pathlib import Path
from typing import Optional

# Theme CSS mapping — maps logical names to CSS files in themes/ dir
THEME_CSS = {
    "analytics": "analytics-light.css",
    "analytics-light": "analytics-light.css",
    "analytics-dark": "analytics-dark.css",
}


def _find_themes_dir(deck_path: Path) -> Optional[Path]:
    """Walk up from deck file looking for a themes/ directory."""
    current = deck_path.resolve().parent
    for _ in range(5):
        candidate = current / "themes"
        if candidate.is_dir():
            return candidate
        current = current.parent
    return None


def _resolve_theme_css(theme: str, deck_path: Path) -> Path:
    """Resolve theme name to CSS file path.

    Args:
        theme: Theme name (e.g. "analytics", "analytics-dark").
        deck_path: Path to the deck file (used to locate themes/ dir).

    Raises:
        FileNotFoundError: If themes/ directory not found.
        ValueError: If theme name is invalid.
    """
    themes_dir = _find_themes_dir(deck_path)
    if themes_dir is None:
        raise FileNotFoundError(
            f"No themes/ directory found near {deck_path}. "
            "Create a themes/ directory with CSS files."
        )

    css_name = THEME_CSS.get(theme)
    if css_name is None:
        raise ValueError(
            f"Unknown theme '{theme}'. Valid: {list(THEME_CSS.keys())}"
        )

    css_path = themes_dir / css_name
    if not css_path.exists():
        raise FileNotFoundError(f"Theme CSS not found: {css_path}")

    return css_path


def _check_marp_cli() -> bool:
    """Check if Marp CLI is available via npx."""
    try:
        result = subprocess.run(
            ["npx", "@marp-team/marp-cli", "--version"],
            capture_output=True, text=True, timeout=15,
        )
        return result.returncode == 0
    except (FileNotFoundError, subprocess.TimeoutExpired):
        return False


def _check_node() -> bool:
    """Check if Node.js is available."""
    try:
        result = subprocess.run(
            ["node", "--version"],
            capture_output=True, text=True, timeout=10,
        )
        return result.returncode == 0
    except (FileNotFoundError, subprocess.TimeoutExpired):
        return False


def _run_marp(
    deck_path: str,
    theme: str = "analytics",
    output_format: str = "pdf",
) -> Path:
    """Run Marp CLI to export a deck.

    Args:
        deck_path: Path to the .marp.md file.
        theme: Theme name.
        output_format: "pdf" or "html".

    Returns:
        Path to the exported file.

    Raises:
        RuntimeError: If Marp CLI fails.
        FileNotFoundError: If deck or theme not found.
    """
    deck = Path(deck_path).resolve()
    if not deck.exists():
        raise FileNotFoundError(f"Deck file not found: {deck}")

    # Determine output path
    stem = deck.stem.replace(".marp", "")
    output_path = deck.parent / f"{stem}.{output_format}"

    # Build command
    cmd = [
        "npx", "@marp-team/marp-cli",
        str(deck),
        f"--{output_format}",
        "--allow-local-files",
        "-o", str(output_path),
    ]

    # Add theme if themes/ dir exists
    try:
        css_path = _resolve_theme_css(theme, deck)
        cmd.extend(["--theme", str(css_path)])
    except (FileNotFoundError, ValueError):
        pass  # Use default Marp theme if custom theme not available

    try:
        result = subprocess.run(
            cmd,
            capture_output=True,
            text=True,
            timeout=120,
            cwd=str(deck.parent),
        )
    except subprocess.TimeoutExpired:
        raise RuntimeError("Marp CLI timed out after 120 seconds")

    if result.returncode != 0:
        raise RuntimeError(f"Marp CLI failed: {result.stderr}")

    return output_path


def export_pdf(deck_path: str, theme: str = "analytics") -> Path:
    """Export a Marp deck to PDF.

    Args:
        deck_path: Path to the .marp.md file.
        theme: Theme name.

    Returns:
        Path to the PDF file.
    """
    return _run_marp(deck_path, theme, "pdf")


def export_html(deck_path: str, theme: str = "analytics") -> Path:
    """Export a Marp deck to HTML.

    Args:
        deck_path: Path to the .marp.md file.
        theme: Theme name.

    Returns:
        Path to the HTML file.
    """
    return _run_marp(deck_path, theme, "html")


def export_both(deck_path: str, theme: str = "analytics") -> dict:
    """Export a Marp deck to both PDF and HTML.

    Args:
        deck_path: Path to the .marp.md file.
        theme: Theme name.

    Returns:
        Dict with "pdf" and "html" paths.
    """
    return {
        "pdf": _run_marp(deck_path, theme, "pdf"),
        "html": _run_marp(deck_path, theme, "html"),
    }


def check_ready() -> dict:
    """Check if the export pipeline is ready.

    Returns:
        Dict with "marp_cli", "node", "themes_available" keys.
    """
    themes_available = []
    # Check from project root
    themes_dir = Path.cwd() / "themes"
    if not themes_dir.is_dir():
        # Try parent dir
        themes_dir = Path.cwd().parent / "themes"

    if themes_dir.is_dir():
        for css_file in themes_dir.glob("*.css"):
            themes_available.append(css_file.stem)

    return {
        "marp_cli": _check_marp_cli(),
        "node": _check_node(),
        "themes_available": themes_available,
    }
