"""The extraction target: one ProductAttributes record per product title.

The same schema is used by the teacher (Claude labels), the student (fine-tuned SLM), guided decoding in vLLM and
the evaluator, so there is exactly one definition of "a valid answer". Labeling rules live in docs/LABELING.md.
"""

from __future__ import annotations

from enum import StrEnum

from pydantic import BaseModel, ConfigDict, Field, field_validator


class Category(StrEnum):
    """Closed top-level taxonomy. Closed so exact match is meaningful and the student can't invent labels."""

    apparel = "apparel"  # clothing, hats, socks, costumes
    footwear = "footwear"
    jewelry_accessories = "jewelry_accessories"  # jewelry, watches, bags, wallets, belts, sunglasses
    beauty_personal_care = "beauty_personal_care"  # cosmetics, skincare, hair care, shaving, oral care
    health_household = "health_household"  # medical supplies, supplements, cleaning products, paper goods
    home_kitchen = "home_kitchen"  # cookware, tableware, drinkware, small kitchen appliances, storage
    home_decor = "home_decor"  # wall art, curtains, rugs, lighting, vases, seasonal decor
    bedding_bath = "bedding_bath"  # sheets, blankets, pillows, towels, shower curtains, bath mats
    furniture = "furniture"
    tools_home_improvement = "tools_home_improvement"  # hand/power tools, hardware, electrical, plumbing
    garden_outdoor = "garden_outdoor"  # planters, patio, lawn care, grills
    automotive = "automotive"  # car/truck/RV/motorcycle parts and accessories
    electronics = "electronics"  # computers, phones, audio, cameras, cables, chargers, car electronics
    sports_fitness = "sports_fitness"  # sports gear, exercise, camping, hunting, fishing, cycling
    toys_games = "toys_games"  # toys, games, puzzles, hobby kits
    baby = "baby"  # baby gear, nursery, diapers, feeding
    pet_supplies = "pet_supplies"
    office_school = "office_school"  # stationery, office supplies, school supplies, printers' consumables
    arts_crafts = "arts_crafts"  # craft supplies, sewing, painting, party supplies
    musical_instruments = "musical_instruments"
    grocery = "grocery"  # food and drink
    industrial_scientific = "industrial_scientific"  # lab, safety, industrial parts
    other = "other"  # gift cards, software, media, anything that fits nowhere else


class ProductAttributes(BaseModel):
    """Attributes extracted from a title alone. Fields are null when the title doesn't state them."""

    model_config = ConfigDict(extra="forbid", use_enum_values=True)

    category: Category = Field(description="Top-level category from the closed taxonomy.")
    brand: str | None = Field(description="Brand name exactly as written in the title, or null.")
    size: str | None = Field(
        description="The product's own primary size/dimension/capacity/weight, copied from the title, or null."
    )
    material: str | None = Field(description="Primary material stated in the title, lowercase, or null.")

    @field_validator("brand", "size", "material", mode="before")
    @classmethod
    def _blank_is_null(cls, v: object) -> object:
        if isinstance(v, str):
            v = " ".join(v.split())
            if v == "" or v.lower() in {"null", "none", "n/a", "unknown"}:
                return None
        return v

    @field_validator("material")
    @classmethod
    def _lowercase_material(cls, v: str | None) -> str | None:
        return v.lower() if v else v


FIELDS: tuple[str, ...] = ("category", "brand", "size", "material")


def json_schema() -> dict:
    """JSON schema for structured outputs / guided decoding."""
    return ProductAttributes.model_json_schema()
