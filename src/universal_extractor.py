"""
Layout-aware medical document extraction.

Preserves text and spatial coordinates so that downstream
extractors can understand tables and multi-column reports.
"""

import pymupdf


def extract_pdf_layout(pdf_path):
    """Extract text lines with their page positions."""
    extracted_lines = []

    with pymupdf.open(pdf_path) as document:
        for page_number, page in enumerate(document, start=1):
            data = page.get_text("dict", sort=True)

            for block in data["blocks"]:
                if "lines" not in block:
                    continue

                for line in block["lines"]:
                    spans = line.get("spans", [])

                    text = " ".join(
                        span["text"].strip()
                        for span in spans
                        if span["text"].strip()
                    )

                    if not text:
                        continue

                    extracted_lines.append({
                        "page": page_number,
                        "text": text,
                        "bbox": list(line["bbox"]),
                    })

    return extracted_lines


def group_layout_rows(lines, y_tolerance=5.0):
    """
    Group nearby text fragments into rows using their
    vertical centers, then order each row left to right.

    This is layout-based, not hospital-specific.
    """
    rows = []

    ordered = sorted(
        lines,
        key=lambda item: (
            item["page"],
            (item["bbox"][1] + item["bbox"][3]) / 2,
        ),
    )

    for item in ordered:
        center_y = (item["bbox"][1] + item["bbox"][3]) / 2

        matching_row = None

        for row in reversed(rows):
            if row["page"] != item["page"]:
                continue

            if abs(row["center_y"] - center_y) <= y_tolerance:
                matching_row = row
                break

        if matching_row is None:
            matching_row = {
                "page": item["page"],
                "center_y": center_y,
                "items": [],
            }
            rows.append(matching_row)

        matching_row["items"].append(item)

        matching_row["center_y"] = sum(
            (cell["bbox"][1] + cell["bbox"][3]) / 2
            for cell in matching_row["items"]
        ) / len(matching_row["items"])

    for row in rows:
        row["items"].sort(key=lambda item: item["bbox"][0])

    return rows



def extract_labs_from_rows(rows):
    """Extract measurements from rows with either value/range column order."""
    import re

    number = r"-?\d+(?:\.\d+)?"
    value_pattern = re.compile(
        rf"^\s*({number})\s*([HL])?\s*$",
        re.IGNORECASE,
    )
    range_pattern = re.compile(
        rf"^\s*({number})\s*[-–]\s*({number})\s*$"
    )

    results = []

    for row in rows:
        cells = [
            item["text"].strip()
            for item in row["items"]
            if item["text"].strip()
        ]

        if len(cells) < 3:
            continue

        for index in range(1, len(cells) - 1):
            first = cells[index]
            second = cells[index + 1]

            value_match = value_pattern.fullmatch(first)
            range_match = range_pattern.fullmatch(second)

            if not (value_match and range_match):
                range_match = range_pattern.fullmatch(first)
                value_match = value_pattern.fullmatch(second)

            if not (value_match and range_match):
                continue

            name = cells[index - 1]

            if not re.search(r"[A-Za-z]", name):
                continue

            value = float(value_match.group(1))
            low = float(range_match.group(1))
            high = float(range_match.group(2))

            if low > high:
                continue

            flag = (value_match.group(2) or "").upper()
            status = (
                "Low" if value < low else
                "High" if value > high else
                "Normal"
            )
            reported_status = {"H": "High", "L": "Low"}.get(flag)

            unit = cells[index + 2] if index + 2 < len(cells) else ""

            results.append({
                "test_name": name,
                "value": value,
                "unit": unit,
                "reference_low": low,
                "reference_high": high,
                "status": status,
                "reported_status": reported_status,
                "page": row["page"],
                "source_text": " | ".join(cells),
                "needs_review": (
                    reported_status is not None
                    and reported_status != status
                ),
            })
            break

    return results
