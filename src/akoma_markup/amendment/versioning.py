"""Version control for amended legislation.

Generates versioned markup files and tracks amendment history.
"""

import json
import logging
from datetime import datetime
from pathlib import Path

logger = logging.getLogger(__name__)


class AmendmentVersion:
    """Represents a single version of legislation."""

    def __init__(
        self,
        act_name: str,
        version_type: str,  # "base" or "amended"
        markup: str,
        metadata: dict | None = None,
    ):
        self.act_name = act_name
        self.version_type = version_type
        self.markup = markup
        self.metadata = metadata or {}
        self.timestamp = datetime.now().isoformat()

    def to_dict(self) -> dict:
        """Convert to dictionary for serialization."""
        return {
            "act_name": self.act_name,
            "version_type": self.version_type,
            "timestamp": self.timestamp,
            "metadata": self.metadata,
        }


def generate_version_filename(
    act_name: str,
    version_type: str,
    timestamp: str | None = None,
) -> str:
    """Generate a standardized filename for a version.

    Args:
        act_name: Name of the act (e.g., "IT Act").
        version_type: "base" or "amended".
        timestamp: Optional timestamp string.

    Returns:
        Filename string.
    """
    safe_name = act_name.lower().replace(" ", "_").replace("(", "").replace(")", "")
    ts = timestamp or datetime.now().strftime("%Y%m%d_%H%M%S")
    return f"{safe_name}_{version_type}_{ts}.txt"


def save_version(
    version: AmendmentVersion,
    output_dir: str | Path,
    filename: str | None = None,
) -> Path:
    """Save a version to disk.

    Args:
        version: AmendmentVersion to save.
        output_dir: Directory to save to.
        filename: Optional custom filename.

    Returns:
        Path to saved markup file.
    """
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    if filename is None:
        filename = generate_version_filename(
            version.act_name,
            version.version_type,
        )

    markup_path = output_dir / filename
    markup_path.write_text(version.markup, encoding="utf-8")

    # Save metadata alongside
    meta_path = markup_path.with_suffix(".meta.json")
    meta_data = version.to_dict()
    meta_data["markup_file"] = str(markup_path)
    meta_path.write_text(json.dumps(meta_data, indent=2), encoding="utf-8")

    logger.info("Saved version to %s", markup_path)
    return markup_path


class AmendmentHistory:
    """Tracks amendment history for an act."""

    def __init__(self, act_name: str, base_version: str):
        self.act_name = act_name
        self.base_version = base_version
        self.amendments: list[dict] = []
        self.versions: list[dict] = []

    def add_amendment(
        self,
        amendment_act: str,
        amendment_year: str,
        sections: list[str],
        effective_date: str | None = None,
    ):
        """Record an amendment in the history."""
        self.amendments.append({
            "amendment_act": amendment_act,
            "amendment_year": amendment_year,
            "sections": sections,
            "effective_date": effective_date,
            "applied_at": datetime.now().isoformat(),
        })

    def add_version(self, version_path: str, version_type: str):
        """Record a version in the history."""
        self.versions.append({
            "path": version_path,
            "type": version_type,
            "timestamp": datetime.now().isoformat(),
        })

    def to_dict(self) -> dict:
        """Convert to dictionary."""
        return {
            "act_name": self.act_name,
            "base_version": self.base_version,
            "amendments": self.amendments,
            "versions": self.versions,
            "generated_at": datetime.now().isoformat(),
        }

    def save(self, output_path: str | Path):
        """Save history to JSON file."""
        output_path = Path(output_path)
        output_path.parent.mkdir(parents=True, exist_ok=True)
        output_path.write_text(
            json.dumps(self.to_dict(), indent=2),
            encoding="utf-8"
        )
        logger.info("Saved amendment history to %s", output_path)


@staticmethod
def load_history(history_path: str | Path) -> "AmendmentHistory":
    """Load amendment history from JSON file."""
    history_path = Path(history_path)
    data = json.loads(history_path.read_text(encoding="utf-8"))

    history = AmendmentHistory(data["act_name"], data["base_version"])
    history.amendments = data.get("amendments", [])
    history.versions = data.get("versions", [])
    return history


def generate_amended_version(
    base_markup: str,
    base_metadata: dict,
    amendments: list[dict],
    output_dir: str | Path,
) -> dict:
    """Generate a new amended version.

    This is a high-level function that:
    1. Saves the base version
    2. Applies amendments and saves amended version
    3. Tracks history

    Args:
        base_markup: Original legislation markup.
        base_metadata: Metadata for base version.
        amendments: List of amendment dicts.
        output_dir: Directory to save versions to.

    Returns:
        Dictionary with paths to saved files.
    """
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    act_name = base_metadata.get("act_name", "Unknown Act")
    base_version = base_metadata.get("base_version", "Unknown")

    history = AmendmentHistory(act_name, base_version)

    # Save base version
    base = AmendmentVersion(
        act_name=act_name,
        version_type="base",
        markup=base_markup,
        metadata=base_metadata,
    )
    base_path = save_version(base, output_dir)
    history.add_version(str(base_path), "base")

    # Record amendments in history
    for amd in amendments:
        history.add_amendment(
            amendment_act=amd["amendment_act"],
            amendment_year=amd.get("amendment_year", ""),
            sections=amd.get("sections", []),
            effective_date=amd.get("effective_date"),
        )

    # Save history
    history_path = output_dir / f"{act_name.lower().replace(' ', '_')}_history.json"
    history.save(history_path)

    return {
        "base_path": base_path,
        "history_path": history_path,
    }
