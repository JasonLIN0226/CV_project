"""Pure parsers for native Slurm text records; no I/O or runtime dependencies.

Every public function accepts only strings. Policy callers supply their own
commands, expected job identity, clocks, and retry behavior.
"""

import re


TERMINAL_STATES = frozenset({"COMPLETED", "CANCELLED", "FAILED", "TIMEOUT", "NODE_FAIL",
                             "PREEMPTED", "BOOT_FAIL", "DEADLINE", "OUT_OF_MEMORY",
                             "REVOKED", "SPECIAL_EXIT"})
STARTUP_STATES = frozenset({"PENDING", "CONFIGURING"})


def _text(value):
    if not isinstance(value, str):
        raise TypeError("Slurm parsers accept strings only")
    return value.strip()


def parse_control(text):
    """Parse one scontrol job record, preserving field values and rejecting ambiguity."""
    text = _text(text)
    fields = {}
    matches = list(re.finditer(r"(?<!\S)([A-Za-z][A-Za-z0-9_/:]*)=", text))
    if not matches or text[:matches[0].start()].strip():
        raise ValueError("Missing or malformed scontrol fields")
    for index, match in enumerate(matches):
        key = match.group(1)
        if key in fields:
            raise ValueError("Duplicate scontrol field: " + key)
        end = matches[index + 1].start() if index + 1 < len(matches) else len(text)
        fields[key] = text[match.end():end].strip()
    return fields


def parse_node_range(value):
    value = _text(value)
    match = re.fullmatch(r"([0-9]+)(?:-([0-9]+))?", value)
    if match is None:
        raise ValueError("Missing or malformed NumNodes")
    minimum = int(match.group(1))
    maximum = int(match.group(2)) if match.group(2) is not None else minimum
    if minimum < 1 or maximum < minimum:
        raise ValueError("Invalid NumNodes range")
    return minimum, maximum


def require_one_node(value):
    if parse_node_range(value) != (1, 1):
        raise ValueError("Native job must request exactly one node")
    return 1


def parse_duration(value):
    value = _text(value)
    match = re.fullmatch(r"(?:(\d+)-)?(\d+):([0-5]\d):([0-5]\d)", value)
    if match is None:
        raise ValueError("Missing, unlimited, or malformed native duration")
    days = int(match.group(1) or 0)
    hours, minutes, seconds = (int(match.group(i)) for i in (2, 3, 4))
    if match.group(1) is not None and hours > 23:
        raise ValueError("Day-qualified native duration has invalid hours")
    return days * 86400 + hours * 3600 + minutes * 60 + seconds


def normalize_state(value):
    value = _text(value)
    if not value:
        raise ValueError("Missing native job state")
    state = value.split()[0].rstrip("+")
    if not re.fullmatch(r"[A-Z][A-Z_]*", state):
        raise ValueError("Malformed native job state")
    return state


def _columns(value):
    names = [name.strip().split("%", 1)[0] for name in value.split(",")]
    if (not names or any(not re.fullmatch(r"[A-Za-z][A-Za-z0-9_]*", name) for name in names)
            or len(set(names)) != len(names)):
        raise ValueError("Missing, malformed, or duplicate accounting columns")
    if sum(name in {"JobIDRaw", "JobID"} for name in names) != 1:
        raise ValueError("Accounting columns must have exactly one job identity column")
    return names


def _cells(line):
    cells = line.split("|")
    if cells and cells[-1] == "":
        cells.pop()
    return [cell.strip() for cell in cells]


def parse_accounting_row(text, job_id, columns_csv=""):
    """Return the exact top-level job row, or None if not yet present.

    Headerless records require comma-separated column names. A pipe-delimited
    header may instead provide the schema. Step rows never substitute for the
    exact requested job, and duplicate or malformed matching rows are errors.
    """
    text, job_id, columns_csv = _text(text), _text(job_id), _text(columns_csv)
    if not re.fullmatch(r"[0-9]+", job_id):
        raise ValueError("Expected an exact numeric top-level Slurm job ID")
    lines = [line.strip() for line in text.splitlines() if line.strip()]
    columns = _columns(columns_csv) if columns_csv else None
    if lines:
        first = _cells(lines[0])
        if "JobIDRaw" in first or "JobID" in first:
            header = _columns(",".join(first))
            if columns is not None and header != columns:
                raise ValueError("Accounting header differs from the requested columns")
            columns = header
            lines = lines[1:]
    if columns is None:
        raise ValueError("Headerless accounting text requires explicit columns")
    identity_index = next(i for i, name in enumerate(columns) if name in {"JobIDRaw", "JobID"})
    selected = []
    for line in lines:
        cells = [cell.strip() for cell in line.split("|")]
        if len(cells) == len(columns) + 1 and cells[-1] == "":
            cells.pop()
        if len(cells) <= identity_index or cells[identity_index] != job_id:
            continue
        if len(cells) != len(columns):
            raise ValueError("Malformed native top-level accounting row")
        selected.append(dict(zip(columns, cells)))
    if len(selected) > 1:
        raise ValueError("Duplicate native top-level accounting rows")
    return selected[0] if selected else None


def _tres_entries(tres):
    tres = _text(tres)
    if tres in {"", "(null)", "None", "Unknown"}:
        return {}
    entries = {}
    for item in tres.split(","):
        if item.count("=") != 1:
            raise ValueError("Malformed native TRES entry")
        key, value = (part.strip() for part in item.split("=", 1))
        if not key or not value or key in entries:
            raise ValueError("Missing or duplicate native TRES entry")
        entries[key] = value
    return entries


def gpu_count(tres):
    """Return 0 or 1 allocated A100 GPUs without double-counting typed TRES."""
    entries = _tres_entries(tres)
    gpu = {key: value for key, value in entries.items()
           if key == "gres/gpu" or key.startswith("gres/gpu:")}
    if not gpu:
        return 0
    if any(not re.fullmatch(r"[0-9]+", value) for value in gpu.values()):
        raise ValueError("Malformed native GPU count")
    typed = [key for key in gpu if key.startswith("gres/gpu:")]
    if len(typed) > 1 or any(key != "gres/gpu:a100" for key in typed):
        raise ValueError("Unexpected native GPU type or multiple typed GPU allocations")
    counts = {int(value) for value in gpu.values()}
    if len(counts) != 1 or not counts <= {0, 1}:
        raise ValueError("Native generic/typed GPU counts disagree or exceed one")
    return counts.pop()


def one_gpu(tres):
    if gpu_count(tres) != 1:
        raise ValueError("Native TRES must certify exactly one A100 GPU")
    return 1
