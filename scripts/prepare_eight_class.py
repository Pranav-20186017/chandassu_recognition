"""Build v3 offline from frozen v2 and explicit archived source annotations.

No rule/model labels, line randomization, OCR or reconstructed missing text.
Seesamu accepts only explicit body+geeti records with 8+4 preserved display lines.
"""

import argparse
import hashlib
import json
import re
import sys
from collections import Counter, defaultdict
from pathlib import Path
from urllib.parse import unquote, urlparse

from bs4 import BeautifulSoup

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))
from chandassu.data import layout_key, normalize_text, work_groups
from chandassu.expanded import LABELS, ExpandedPoem, assert_parts, counts
from chandassu.io import atomic_json, identity, sha256

ALIASES = {"ఆ": "ఆటవెలది", "ఆవె": "ఆటవెలది", "తే": "తేటగీతి", "క": "కందము"}
ROOT = Path(__file__).resolve().parents[1]


def read(path):
    return [json.loads(line) for line in path.open() if line.strip()]


def marker(row):
    raw = row.get("raw_marker") or next(iter(re.findall(r"Printed marker '([^']+)'", row.get("label_notes", ""))), "")
    return re.sub(r"\[[^]]*\]", "", raw), raw


def code(value):
    return "".join(c for c in value if "\u0c00" <= c <= "\u0c7f" and not c.isdigit())


def build(base, output):
    if output.exists() and any(output.iterdir()):
        raise ValueError("Output exists; choose a new version instead of overwriting frozen data")
    output.mkdir(parents=True, exist_ok=True)
    inputs = {}

    def load(path):
        inputs[str(path.relative_to(ROOT))] = sha256(path)
        return read(path)

    partition = json.loads((base / "splits/train_expansion.json").read_text())["ids"]
    inputs[str((base / "manifest.json").relative_to(ROOT))] = sha256(base / "manifest.json")
    inputs[str((base / "splits/train_expansion.json").relative_to(ROOT))] = sha256(base / "splits/train_expansion.json")
    id_role = {pid: role for role, ids in partition.items() for pid in ids}
    old = []
    for role in ["train", "validation", "test"]:
        for row in load(base / f"{role}.jsonl"):
            row["frozen_role"] = "test" if role == "test" else id_role[row["poem_id"]]
            row["cohort"] = "v2_existing"
            old.append(row)
    source_alias = {layout_key(r["source"]): r["source"] for r in old}
    known_editions = {r["source"]: r.get("edition_family") or r["source"] for r in old}
    candidates, exclusions = [], []
    inventory = Counter()
    raw_paths = [ROOT / "data/v1/other.jsonl", ROOT / "archive/data/expansion_20261011/extraction.jsonl"]
    # The canonical v1 other file already includes accepted independent v1/v2 extractions.
    for path in raw_paths:
        for row in load(path):
            clean_marker, raw_marker = marker(row)
            label = ALIASES.get(code(clean_marker))
            if not label:
                if row.get("domain") == "other":
                    inventory[f"ambiguous_or_other:{clean_marker}"] += 1
                continue
            inventory[label] += 1
            row = dict(row, label=label, raw_marker=raw_marker, import_file=str(path.relative_to(ROOT)))
            row.pop("role", None)
            candidates.append(row)
    for batch in ["independent_poetry/v1", "independent_poetry/v2", "expansion_20261011"]:
        path = ROOT / "archive/data" / batch / "review.jsonl"
        for row in load(path):
            m, raw = marker(row)
            if not m.startswith("సీ"):
                continue
            inventory["seesamu_review_records"] += 1
            # A compound marker alone with nonstandard boundaries is insufficient.
            chunks = m.split("+")
            if len(chunks) != 2 or code(chunks[0]) != "సీ" or code(chunks[1]) not in {"ఆ", "ఆవె", "తే", "గీ"}:
                exclusions.append(
                    {"source": row["source"], "raw_marker": raw, "reason": "seesamu_missing_explicit_closure"}
                )
                continue
            lines = row.get("lines", [])
            if len(lines) != 12 or set(row.get("reasons", [])) - {"compound_metre_component", "line_boundary_review"}:
                exclusions.append(
                    {"source": row["source"], "raw_marker": raw, "reason": "seesamu_ambiguous_boundaries"}
                )
                continue
            inventory["seesamu_explicit_8_plus_4"] += 1
            pid = "seesamu:" + identity([unquote(row["source_url"]), row["location"]])[:24]
            html = path.parent / row["input_file"]
            candidates.append(
                dict(
                    row,
                    poem_id=pid,
                    label="సీసము",
                    raw_marker=raw,
                    raw_lines=lines,
                    lines=[" ".join(lines[i : i + 2]) for i in range(0, 8, 2)],
                    closing_geeti_lines=lines[8:],
                    closing_geeti_marker=chunks[1],
                    representation="four ordered body padas paired from eight explicit half-lines; closure excluded",
                    label_status="source_annotation",
                    label_origin="source",
                    audit_status="source_annotation_unverified",
                    raw_html_file=str(html.relative_to(ROOT)),
                    raw_html_sha256=sha256(html),
                    import_file=str(path.relative_to(ROOT)),
                )
            )
    # Recover explicitly annotated non-vrutta verse cells that the original
    # four-class AndhraBharati scraper skipped, using its saved page evidence.
    for report_path in sorted((ROOT / "archive/data/andhrabharati").glob("*/report.json")):
        report = json.loads(report_path.read_text())
        if not isinstance(report.get("pages"), list) or report.get("work_key") == "manu-pilot":
            continue
        inputs[str(report_path.relative_to(ROOT))] = sha256(report_path)
        for page in report.get("pages", []):
            path = report_path.parent / "raw" / Path(urlparse(page["url"]).path).name
            if not path.exists():
                continue
            digest = sha256(path)
            normalized_digest = hashlib.sha256(path.read_text(encoding="utf-8-sig").encode()).hexdigest()
            if page["raw_html_sha256"] not in {digest, normalized_digest}:
                exclusions.append(
                    dict(source=report["source"], reason="archived_html_checksum_changed", path=str(path))
                )
                continue
            inputs[str(path.relative_to(ROOT))] = digest
            soup = BeautifulSoup(path.read_text(), "html.parser")
            for n, tr in enumerate(soup.select("div.wmsect tr"), 1):
                cells = tr.find_all("td", recursive=False)
                if len(cells) != 3 or any(c.find("table") for c in cells):
                    continue
                raw = cells[0].get_text(" ", strip=True)
                label = ALIASES.get(code(raw))
                if label is None:
                    continue
                inventory["raw_html:" + label] += 1
                cell = BeautifulSoup(str(cells[1]), "html.parser")
                for ref in cell.select("sup, script, style"):
                    ref.decompose()
                for br in cell.find_all("br"):
                    br.replace_with("\x00")
                lines = [normalize_text(t) for t in cell.get_text().split("\x00") if normalize_text(t)]
                if len(lines) != 4:
                    exclusions.append(dict(source=report["source"], reason="raw_html_nonfour_boundaries"))
                    continue
                candidates.append(
                    dict(
                        poem_id="ab-eight:" + identity([page["url"], n])[:24],
                        source=report["source"],
                        author=report["author"] or "అజ్ఞాతం",
                        label=label,
                        lines=lines,
                        raw_lines=lines,
                        raw_marker=raw,
                        source_url=page["url"],
                        label_status="source_annotation",
                        label_origin="source",
                        audit_status="source_annotation_unverified",
                        raw_html_file=str(path.relative_to(ROOT)),
                        raw_html_sha256=digest,
                        import_file=str(report_path.relative_to(ROOT)),
                    )
                )
    unique = {}
    old_keys = {tuple(layout_key(t) for t in r["lines"]) for r in old}
    for row in candidates:
        try:
            poem = ExpandedPoem.from_dict(row)
            if any(not layout_key(t) or re.search(r"\.{3}|…|[A-Za-z]|[<>]", t) for t in poem.lines):
                raise ValueError("lacuna")
        except ValueError:
            exclusions.append({"source": row["source"], "reason": "invalid_or_lacuna"})
            continue
        key = tuple(layout_key(t) for t in poem.lines)
        if key in old_keys or key in unique:
            if key in unique and unique[key]["label"] != row["label"]:
                raise ValueError("Conflicting explicit labels for duplicate poem")
            exclusions.append({"source": row["source"], "reason": "duplicate_poem"})
            continue
        row["source"] = source_alias.get(layout_key(row["source"]), row["source"])
        row["edition_family"] = known_editions.get(row["source"], row.get("edition_family") or row["source"])
        row["lines"] = list(poem.lines)
        row["cohort"] = "new_source_annotation"
        row["domain"] = "target"
        unique[key] = row
    candidates = list(unique.values())
    old_source_roles = {r["source"]: r["frozen_role"] for r in old}
    pinned_line_roles = defaultdict(set)
    for row in old + candidates:
        role = row.get("frozen_role") or old_source_roles.get(row["source"])
        if role:
            for text in row["lines"] + row.get("closing_geeti_lines", []):
                pinned_line_roles[layout_key(text)].add(role)
    retained = []
    for row in candidates:
        expected_role = old_source_roles.get(row["source"])
        linked_roles = set().union(
            *(pinned_line_roles[layout_key(t)] for t in row["lines"] + row.get("closing_geeti_lines", []))
        )
        if len(linked_roles) > 1 or (expected_role and linked_roles - {expected_role}):
            exclusions.append(dict(source=row["source"], poem_id=row["poem_id"], reason="cross_role_quotation"))
        else:
            retained.append(row)
    candidates = retained
    all_rows = old + candidates
    poems = [ExpandedPoem.from_dict(row) for row in all_rows]
    # Include Seesamu closure lines in duplicate/work grouping, even though not model inputs.
    grouping = [
        ExpandedPoem(
            p.poem_id,
            p.source,
            p.author,
            p.lines + tuple(p.metadata.get("closing_geeti_lines", [])),
            p.label,
            p.metadata,
        )
        for p in poems
    ]
    groups = work_groups(grouping)
    fixed = defaultdict(set)
    for row, group in zip(all_rows, groups):
        if row.get("frozen_role"):
            fixed[group].add(row["frozen_role"])
    conflicts = {g for g, roles in fixed.items() if len(roles) > 1}
    if conflicts:
        # New quotation/edition bridges must not merge or reassign frozen historical works.
        bad_sources = {
            row["source"]
            for row, g in zip(all_rows, groups)
            if g in conflicts and row["source"] not in old_source_roles
        }
        candidates = [r for r in candidates if r["source"] not in bad_sources]
        exclusions.extend({"source": s, "reason": "bridges_frozen_roles"} for s in sorted(bad_sources))
        all_rows = old + candidates
        poems = [ExpandedPoem.from_dict(r) for r in all_rows]
        grouping = [
            ExpandedPoem(
                p.poem_id,
                p.source,
                p.author,
                p.lines + tuple(p.metadata.get("closing_geeti_lines", [])),
                p.label,
                p.metadata,
            )
            for p in poems
        ]
        groups = work_groups(grouping)
        fixed = defaultdict(set)
        for row, g in zip(all_rows, groups):
            if row.get("frozen_role"):
                fixed[g].add(row["frozen_role"])
        if any(len(v) > 1 for v in fixed.values()):
            raise ValueError("Remaining historical group conflict requires audit")
    grouped = defaultdict(list)
    for row, g in zip(all_rows, groups):
        grouped[g].append(row)
    allocation = {g: next(iter(roles)) for g, roles in fixed.items() if roles}
    # Deterministic count balancing of whole NEW groups, solely on label counts.
    # Existing roles remain pinned. A fixed seed/hash breaks ties; no model scores.
    fractions = {"fit": 0.70, "selection": 0.15, "test": 0.15}
    new_labels = LABELS[4:]
    totals = Counter(r["label"] for r in candidates)
    current = {role: Counter() for role in fractions}
    for g, role in allocation.items():
        if role in current:
            current[role].update(r["label"] for r in grouped[g] if r["label"] in new_labels)
    pending = sorted((g for g in grouped if g not in allocation), key=lambda g: (-len(grouped[g]), identity([142, g])))
    for g in pending:
        c = Counter(r["label"] for r in grouped[g])

        def cost(role):
            return sum(
                ((current[rr][lab] + (c[lab] if rr == role else 0) - frac * totals[lab]) / max(1, frac * totals[lab]))
                ** 2
                for rr, frac in fractions.items()
                for lab in new_labels
            )

        chosen = min(fractions, key=cost)
        allocation[g] = chosen
        current[chosen].update(c)
    parts = {role: [] for role in ["fit", "selection", "calibration", "test"]}
    records = {role: [] for role in parts}
    for row, group in zip(all_rows, groups):
        role = allocation[group]
        row["group_id"] = identity(group)[:20]
        records[role].append(row)
        parts[role].append(ExpandedPoem.from_dict(row))
    assert_parts(parts)
    files = {}
    for role, rows in records.items():
        p = output / f"{role}.jsonl"
        p.write_text(
            "".join(
                json.dumps(r, ensure_ascii=False, sort_keys=True) + "\n"
                for r in sorted(rows, key=lambda r: r["poem_id"])
            )
        )
        files[p.name] = sha256(p)
    catalogue = []
    md = [
        "# Eight-class work and source catalogue",
        "",
        "V2 roles are frozen. Seesamu counts refer to body-only inputs; closure text stays attached as provenance.",
        "",
        "| Work | Author | Partition | Poems | Labels | Sources |",
        "|---|---|---|---:|---|---|",
    ]
    for source in sorted({r["source"] for r in all_rows}):
        rows = [r for r in all_rows if r["source"] == source]
        role = allocation[groups[next(i for i, r in enumerate(all_rows) if r["source"] == source)]]
        urls = sorted({r["source_url"] for r in rows if r.get("source_url")})
        entry = dict(
            work=source,
            authors=sorted({r["author"] for r in rows}),
            partition=role,
            poems=len(rows),
            labels=dict(Counter(r["label"] for r in rows)),
            source_urls=urls,
            edition_family=rows[0]["edition_family"],
            rights_status="Source rights apply; not independently cleared",
        )
        catalogue.append(entry)
        md.append(
            f"| {source} | {'; '.join(entry['authors'])} | {role} | {len(rows)} | {', '.join(entry['labels'])} | "
            + " ".join(f"[source {i + 1}]({url})" for i, url in enumerate(urls))
            + " |"
        )
    atomic_json(output / "works.json", catalogue)
    (output / "WORKS.md").write_text("\n".join(md) + "\n")
    atomic_json(
        output / "audit.json",
        dict(
            input_files=inputs,
            raw_inventory=dict(inventory),
            exclusions=dict(Counter(r["reason"] for r in exclusions)),
            exclusion_records=exclusions,
            near_duplicate_screen="Exact layout-equivalent lines and poem keys; fuzzy/paraphrase overlap not established",
            labels_are_expert_gold=False,
            test_policy="V2 historical benchmark plus previously untrained new-label work holdouts",
        ),
    )
    for name in ["works.json", "WORKS.md", "audit.json"]:
        files[name] = sha256(output / name)
    manifest = dict(
        schema_version=3,
        labels=list(LABELS),
        representation="four ordered padas; Seesamu body only",
        counts={role: counts(rows) for role, rows in parts.items()},
        files=files,
        split_policy="Frozen v2 roles; new whole work/edition/exact-duplicate groups, 70/15/15 target",
        split_seed=142,
        inputs=inputs,
    )
    atomic_json(output / "manifest.json", manifest)
    print(
        json.dumps(
            dict(
                counts=manifest["counts"],
                raw_inventory=dict(inventory),
                exclusions=Counter(r["reason"] for r in exclusions),
            ),
            ensure_ascii=False,
            indent=2,
        )
    )


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--base", type=Path, default=ROOT / "data/v2")
    parser.add_argument("--output", type=Path, default=ROOT / "data/v3")
    args = parser.parse_args()
    build(args.base, args.output)
