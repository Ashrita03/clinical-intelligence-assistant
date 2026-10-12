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
    """Extract laboratory results from variable column layouts."""
    import re

    number = r"-?\d+(?:\.\d+)?"

    value_pattern = re.compile(
        rf"^\s*({number})\s*([HL])?\s*$",
        re.IGNORECASE,
    )
    value_unit_pattern = re.compile(
        rf"^\s*({number})\s+([^\d\s][^\s]*)\s*$"
    )
    interval_pattern = re.compile(
        rf"^\s*({number})\s*[-–]\s*({number})\s*$"
    )
    upper_pattern = re.compile(
        rf"^\s*(?:up\s*to|<=|≤|<)\s*({number})\s*$",
        re.IGNORECASE,
    )
    lower_pattern = re.compile(
        rf"^\s*(?:>=|≥|>)\s*({number})\s*$"
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

        candidates = []

        for index, cell in enumerate(cells):
            interval = interval_pattern.fullmatch(cell)
            upper = upper_pattern.fullmatch(cell)
            lower = lower_pattern.fullmatch(cell)

            if interval:
                candidates.append((
                    index,
                    float(interval.group(1)),
                    float(interval.group(2)),
                ))
            elif upper:
                candidates.append((index, None, float(upper.group(1))))
            elif lower:
                candidates.append((index, float(lower.group(1)), None))

        for range_index, low, high in candidates:
            if low is not None and high is not None and low > high:
                continue

            for value_index in range(1, len(cells)):
                if value_index == range_index:
                    continue

                match = value_pattern.fullmatch(cells[value_index])
                combined = value_unit_pattern.fullmatch(cells[value_index])

                if match:
                    value = float(match.group(1))
                    flag = (match.group(2) or "").upper()
                    embedded_unit = ""
                elif combined:
                    value = float(combined.group(1))
                    flag = ""
                    embedded_unit = combined.group(2)
                else:
                    continue

                name = cells[0]
                if not re.search(r"[A-Za-z]", name):
                    continue

                unit = embedded_unit

                if not unit:
                    for unit_index in range(1, len(cells)):
                        if unit_index in (value_index, range_index):
                            continue
                        candidate = cells[unit_index]
                        if (
                            re.fullmatch(r"[^\d\s]+", candidate)
                            and candidate.upper() not in ("H", "L")
                        ):
                            unit = candidate
                            break

                status = "Normal"
                if low is not None and value < low:
                    status = "Low"
                elif high is not None and value > high:
                    status = "High"

                reported_status = {
                    "H": "High",
                    "L": "Low",
                }.get(flag)

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

            else:
                continue
            break

    return results
