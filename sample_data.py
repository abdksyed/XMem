#!/usr/bin/env python3
"""Create balanced patient-subset and frame-sampling manifests for EndoVis."""

import argparse
import itertools
import json
import random
from pathlib import Path

from PIL import Image


DATASETS = {
    "endovis17": {"patients": [1, 2, 4, 8], "frames": [30, 50, 100, 225]},
    "endovis18": {"patients": [1, 2, 4, 8, 15], "frames": [30, 50, 100, 300]},
}


def first_annotated(mask_files):
    for i, filename in enumerate(mask_files):
        with Image.open(filename) as mask:
            if any(mask.getdata()):
                return i
    raise ValueError(f"No annotated frame found in {mask_files[0].parent}")


def evenly_spaced(files, count, first_index):
    files = files[first_index:]
    if len(files) < count:
        raise ValueError(
            f"{files[0].parent} has only {len(files)} annotated frames; "
            f"cannot select {count}"
        )
    if count == 1:
        return [files[0]]
    indices = [round(i * (len(files) - 1) / (count - 1)) for i in range(count)]
    return [files[i] for i in indices]


def balanced_subsets(patients, size, count, rng):
    all_subsets = list(itertools.combinations(patients, size))
    if count >= len(all_subsets):
        return [list(x) for x in all_subsets]

    # Greedily choose random subsets containing patients with the fewest uses.
    # This keeps patient representation as even as possible.
    chosen = []
    uses = {patient: 0 for patient in patients}
    for _ in range(count):
        remaining = [x for x in all_subsets if x not in chosen]
        def score(candidate):
            next_uses = uses.copy()
            for patient in candidate:
                next_uses[patient] += 1
            return (max(next_uses.values()), sum(next_uses.values()))

        best_score = min(score(candidate) for candidate in remaining)
        candidates = [x for x in remaining if score(x) == best_score]
        subset = list(rng.choice(candidates))
        chosen.append(subset)
        for patient in subset:
            uses[patient] += 1
    return chosen


def link(source, destination):
    destination.parent.mkdir(parents=True, exist_ok=True)
    if destination.exists() or destination.is_symlink():
        destination.unlink()
    destination.symlink_to(source.resolve())


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--dataset", choices=DATASETS, required=True)
    parser.add_argument("--frames", type=Path, required=True)
    parser.add_argument("--masks", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument(
        "--test-patients", nargs="+", required=True,
        help="Patient/sequence directory names reserved for testing",
    )
    parser.add_argument("--seed", type=int, default=2025)
    args = parser.parse_args()

    rng = random.Random(args.seed)
    test_patients = set(args.test_patients)
    frame_root = args.frames
    mask_root = args.masks

    patients = sorted(
        x.name for x in frame_root.iterdir()
        if x.is_dir() and x.name not in test_patients
    )
    missing_masks = [p for p in patients if not (mask_root / p).is_dir()]
    if missing_masks:
        raise ValueError(f"Missing mask directories: {missing_masks}")

    expected = DATASETS[args.dataset]["patients"]
    if len(patients) != max(expected):
        raise ValueError(
            f"Expected {max(expected)} training patients after excluding tests, "
            f"found {len(patients)}: {patients}"
        )

    manifest = {
        "dataset": args.dataset,
        "seed": args.seed,
        "test_patients": sorted(test_patients),
        "training_patients": patients,
        "experiments": [],
    }

    for patient_count in expected:
        subset_count = 1 if patient_count == len(patients) else min(
            30, len(list(itertools.combinations(patients, patient_count)))
        )
        subsets = balanced_subsets(patients, patient_count, subset_count, rng)

        for subset_number, subset in enumerate(subsets, start=1):
            for frame_count in DATASETS[args.dataset]["frames"]:
                experiment = f"p{patient_count:02d}_s{subset_number:02d}_f{frame_count}"
                record = {
                    "name": experiment,
                    "patients": subset,
                    "frames_per_patient": frame_count,
                    "frames": {},
                }

                for patient in subset:
                    source_frames = sorted((frame_root / patient).iterdir())
                    source_masks = {x.name: x for x in (mask_root / patient).iterdir()}
                    common = [x for x in source_frames if x.name in source_masks]
                    common.sort()
                    start = first_annotated([source_masks[x.name] for x in common])
                    selected = evenly_spaced(common, frame_count, start)
                    record["frames"][patient] = [x.name for x in selected]

                    for source_frame in selected:
                        relative = Path(experiment) / patient / source_frame.name
                        link(source_frame, args.output / "frames" / relative)
                        link(source_masks[source_frame.name], args.output / "masks" / relative)

                manifest["experiments"].append(record)

    args.output.mkdir(parents=True, exist_ok=True)
    (args.output / "manifest.json").write_text(json.dumps(manifest, indent=2))
    print(f"Wrote {len(manifest['experiments'])} experiments to {args.output}")


if __name__ == "__main__":
    main()
