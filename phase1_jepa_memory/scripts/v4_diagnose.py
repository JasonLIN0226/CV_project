#!/usr/bin/env python3
"""Run the predeclared frozen-model diagnostics without modifying v3 artifacts."""
import argparse
from pathlib import Path

from phase1.gate_diagnostics import export_gate_pairs


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--development-output", required=True)
    parser.add_argument("--output", required=True)
    args = parser.parse_args()
    source, destination = Path(args.development_output).resolve(), Path(args.output).resolve()
    export_gate_pairs(source, destination)
    from phase1.diagnostics import motion_cross_policy
    motion_cross_policy(source / "motion" / "reference", source / "motion" / "motion_wide_span", destination, device="cpu")


if __name__ == "__main__":
    main()
