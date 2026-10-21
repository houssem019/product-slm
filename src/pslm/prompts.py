"""Prompts. There are two on purpose:

* TEACHER: the full labeling guidelines (docs/LABELING.md) plus worked examples. It's long (~2k tokens), but it's
  cached, and a frontier or zero-shot model needs the rules spelled out to label consistently.
* STUDENT: one line plus the category list. The fine-tuned SLM learns the rules from the labels, so a short
  prompt keeps training sequences and inference latency small. Baselines are never run with it, because they
  were never taught the rules.
"""

from __future__ import annotations

import json
from functools import cache

from pslm.config import ROOT
from pslm.schema import Category

GUIDELINES = ROOT / "docs" / "LABELING.md"
CATEGORIES = ", ".join(c.value for c in Category)

# Worked examples, verbatim train titles with their labels (tests assert they never come from val/test). They
# show the rules that are easiest to get wrong: a brand that isn't first, compatibility sizes, plating.
EXAMPLES = [
    (
        "ICON SERIES BY THERMOS Stainless Steel Water Bottle with Spout 24 Ounce, Glacier",
        {"category": "home_kitchen", "brand": "THERMOS", "size": "24 Ounce", "material": "stainless steel"},
    ),
    (
        "GMJzzx Designed for iPhone 14 Pro Max Case 6.7 Inch,Full Body Protective Shockproof Hard Back Woman "
        "Girls, Dual Layer Heavy Duty Bumper Cover for iPhone 14 Pro Max Baseball on fire and Water",
        {"category": "electronics", "brand": "GMJzzx", "size": None, "material": None},
    ),
    (
        "14K Gold Plated Animal Dog Pendant Necklace for Women and Men, Cubic Zirconia Necklace for Dog Lovers "
        'and Dog Owners, 18" Chain',
        {"category": "jewelry_accessories", "brand": None, "size": '18"', "material": None},
    ),
    (
        "Brock Replacement Driver and Passenger Side Chrome Power Mirrors Manual Foldaway without Heat or Signal "
        "with Adapter Compatible with 1997-2003 F-150/2004 F-150 Heritage/ 1997-1999 F-250 Light Duty",
        {"category": "automotive", "brand": "Brock", "size": None, "material": None},
    ),
]


@cache
def teacher_system() -> str:
    examples = "\n".join(f"Title: {t}\nJSON: {json.dumps(a)}" for t, a in EXAMPLES)
    return (
        "You label retail product titles with structured attributes for a training dataset. Follow the "
        "guidelines below exactly; consistency matters more than cleverness.\n\n"
        f"<guidelines>\n{GUIDELINES.read_text(encoding='utf-8').strip()}\n</guidelines>\n\n"
        f"Allowed categories: {CATEGORIES}.\n\n"
        f"<examples>\n{examples}\n</examples>\n\n"
        'Reply with only a JSON object with the keys "category", "brand", "size" and "material". '
        "Use null for any attribute the title does not state."
    )


STUDENT_SYSTEM = (
    "Extract product attributes from the title as JSON with keys category, brand, size, material "
    f"(null if not stated). Categories: {CATEGORIES}."
)


def user_message(title: str) -> str:
    return f"Title: {title}"
