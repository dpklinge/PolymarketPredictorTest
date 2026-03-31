from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path


@dataclass
class CategoryTaxonomy:
    preferred_categories: set[str]
    generic_tags: set[str]
    aliases: dict[str, str]

    def normalize(self, raw: str) -> str:
        value = raw.strip().lower()
        if not value:
            return ""
        return self.aliases.get(value, value)


def load_taxonomy(path: str | Path | None = None) -> CategoryTaxonomy:
    taxonomy_path = Path(path) if path else Path(__file__).with_name("taxonomy.json")
    payload = json.loads(taxonomy_path.read_text(encoding="utf-8"))
    return CategoryTaxonomy(
        preferred_categories={item.strip().lower() for item in payload.get("preferred_categories", [])},
        generic_tags={item.strip().lower() for item in payload.get("generic_tags", [])},
        aliases={key.strip().lower(): value.strip().lower() for key, value in payload.get("aliases", {}).items()},
    )
