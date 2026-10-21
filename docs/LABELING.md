# Labeling guidelines

The model sees **only the title**. It must not use the seller/store name or outside knowledge of the product.
These rules apply to the teacher (Claude), the human review of the test set, and the evaluator. Schema:
`src/pslm/schema.py`.

The rules are written to remove ambiguity. Two careful annotators should produce the same label. When the rules
don't settle a case, pick the more conservative answer, which usually means `null`.

## category: required, one of the closed taxonomy
- Label **what the product is**, not who it's for or which theme it carries:
  - an NCAA dog collar is `pet_supplies`;
  - a car phone mount is `automotive`;
  - a baby-animal shower curtain is `bedding_bath`.
- Kitchen tools and drinkware are `home_kitchen`, even when they're themed or meant for kids.
- **Indoor home appliances are `home_kitchen`:** vacuums, fans, portable heaters and air coolers. Outdoor heaters are `garden_outdoor`.
- **Clothing is always `apparel`**, including baby clothes. `baby` covers gear, nursery items, feeding and breast pumps.
- **Video-game accessories, phone cases and chargers are `electronics`.** Car-mounted gadgets are `automotive`.
- **Greeting cards, calendars and planners are `office_school`.** Party decorations are `arts_crafts`.
- **Gloves follow their purpose:** work and safety gloves are `industrial_scientific`; warm or fashion gloves are `apparel`.
- **When the title gives no clue** (e.g. "Pomchies unisex-adult Modern"), pick the most likely category from the words that are there.
- Use `other` only for gift cards, software, media, and items that truly fit nowhere else.

## brand: string or null
- Use the brand **only if it appears in the title**. Copy it exactly as written, keeping its casing:
  - `JHYWELL`, `Koala Baby`, `Chuck E. Cheese`.
- The brand is usually the first token(s). A brand that appears later counts too, e.g. "Replacement filter by Brita".
- **Not a brand:**
  - **Compatibility targets:** "Replaces John Deere", "for iPhone 15", "Compatible with Dyson".
  - **Licensed themes or teams:** NCAA, Disney, Marvel, unless they are clearly the maker.
  - **Generic descriptors:** "Premium", "Upgraded", "2024 New".
- If the title has no brand, use `null`. This is common for generic marketplace titles.

## size: string or null
- This is the product's **own** primary size: a dimension, length, capacity, volume, weight, or garment/shoe size.
- **Copy the span from the title exactly**, e.g. `89x63cm`, `20 oz`, `XL`, `6 ft`, `Queen`.
  - If the title gives several units for the same size (`16.4ft/5m`), take the first: `16.4ft`.
  - If it lists several sizes or variants, take the first one mentioned. That includes named sizes
    (`Regular & Plus Size` → `Regular`, `Junior Size, 5-1/2" x 8-1/2"` → `Junior`).
  - Copy only the value, without a label: `Height = 5/8"` → `5/8"`.
- Ambiguous bare numbers (`Willow Bucket 24`) and optics specs (`18-200mm` focal length) are `null`.
- **Not a size:**
  - pack counts or piece counts (`5 Pcs`, `Pack of 2`);
  - electrical ratings (`1000V 40A`, `60W`);
  - screen resolutions;
  - ages (`3+ years`);
  - sizes of a *compatible* item (`for 42 inch deck`, `fits 55" TV`).
- If there's no size, use `null`.
- The evaluator compares sizes case-insensitively and ignores spaces, so `20 oz` and `20oz` count as equal.

## material: string or null
- This is the primary material of the product **as stated** in the title, written in lowercase:
  - `stainless steel`, `cotton`, `acrylic`, `pu leather`, `silicone`.
- If several materials are listed, take the one that makes up the main body. If that's unclear, take the first one listed.
- **Never infer:** "Coffee Mug" alone is `null`, not `ceramic`.
- **Normalize the wording:**
  - Use the noun: `Wooden` → `wood`.
  - Drop marketing qualifiers: `Real Leather` → `leather`.
  - Keep real compound names: `stainless steel`, `merino wool`, `organic cotton`, `14k yellow gold`, `pu leather`.
  - Keep abbreviations as written: `PC Case` → `pc`.
- **Plating is a finish, not a material:** `18K Gold Plated` and `Brass Finish` are `null`.
- **Ignore secondary parts:** a lining ("Satin Lined"), the container of a liquid, or a material named only for an accessory.
- **Not a material:** finishes and colors (`matte black`, `gold`), and fill power or weave names when no material is given.
