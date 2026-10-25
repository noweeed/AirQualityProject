def md_table(rows, columns):
    """Build a markdown table by hand from a list of dicts (no extra packages)."""
    def cell(value):
        # A "|" inside a cell would break the table, so escape it
        return str(value).replace("|", "\\|")

    lines = [
        "| " + " | ".join(columns) + " |",
        "|" + "|".join("---" for _ in columns) + "|",
    ]
    for row in rows:
        lines.append("| " + " | ".join(cell(row[c]) for c in columns) + " |")
    return "\n".join(lines) + "\n"
