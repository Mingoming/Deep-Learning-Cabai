"""Persiapan dataset satu kali; notebook training memakai artefak hasilnya."""

import csv
import hashlib
import json
import os
import shutil
import stat
import tempfile
import zipfile
from collections import Counter, defaultdict
from pathlib import Path, PurePosixPath

from PIL import Image

PROJECT_ROOT = Path(__file__).resolve().parents[1]
DATA_DIR = Path(os.environ.get("DL_CABAI_DATA_DIR", PROJECT_ROOT / "data")).expanduser().resolve()
SOURCE_ARCHIVE = Path(os.environ.get(
    "DL_CABAI_ARCHIVE",
    DATA_DIR / "Chilli Leaf Disease Image Dataset for Classificati.zip",
)).expanduser().resolve()
EXTRACTED_DIR = DATA_DIR / "extracted"
PROCESSED_DIR = DATA_DIR / "processed"

CLASS_DIR_TO_LABEL = {
    "Bacterial_Spot": "Bacterial Spot",
    "Cercospora_Leaf_Spot": "Cercospora Leaf Spot",
    "Curl_Virus": "Curl Virus",
    "Healthy_Leaf": "Healthy Leaf",
    "Nutrition_Deficiency": "Nutrient Deficiency",
    "Powdery_Mildew": "Powdery Mildew",
}
CLASS_LABELS = tuple(CLASS_DIR_TO_LABEL.values())
SPLITS = ("train", "validation", "test")
SPLIT_RATIOS = {"train": 0.70, "validation": 0.15, "test": 0.15}
DEFAULT_SEEDS = (42, 43, 44)
IMAGE_SIZE = (224, 224)
NEAR_DUPLICATE_HAMMING_THRESHOLD = 4
MANIFEST_FIELDS = (
    "relative_path", "class_name", "seed", "split", "status", "duplicate_of",
    "near_duplicate_group", "sha256", "perceptual_hash", "extension", "format",
    "width", "height", "mode", "file_size_bytes", "error",
)


def safe_parts(name):
    if not name or "\\" in name or "\x00" in name:
        raise ValueError(f"Path ZIP tidak aman: {name!r}")
    path = PurePosixPath(name.rstrip("/"))
    if not path.parts or path.is_absolute() or any(":" in part for part in path.parts) or ".." in path.parts:
        raise ValueError(f"Path ZIP tidak aman: {name!r}")
    return path.parts


def write_member(archive, info, root, relative_parts):
    target = root.joinpath(*relative_parts).resolve()
    if root.resolve() not in target.parents:
        raise ValueError(f"Path keluar dari folder ekstraksi: {info.filename}")
    if stat.S_ISLNK((info.external_attr >> 16) & 0o170000):
        raise ValueError(f"Symlink ZIP tidak diizinkan: {info.filename}")
    if target.exists():
        raise ValueError(f"Path anggota ZIP ganda: {target}")
    target.parent.mkdir(parents=True, exist_ok=True)
    with archive.open(info) as source, target.open("xb") as output:
        shutil.copyfileobj(source, output)


def extract_members(archive, root, strip_first=None):
    root.mkdir(parents=True, exist_ok=True)
    for info in archive.infolist():
        if info.is_dir():
            continue
        parts = safe_parts(info.filename)
        if strip_first and parts[0] == strip_first:
            parts = parts[1:]
        if not parts:
            raise ValueError(f"Path anggota kosong: {info.filename}")
        write_member(archive, info, root, parts)


def safe_extract_dataset(archive_path, destination):
    archive_path, destination = Path(archive_path), Path(destination)
    if not archive_path.is_file():
        raise FileNotFoundError(archive_path)
    if destination.exists():
        raise FileExistsError(f"Tidak menimpa folder yang sudah ada: {destination}")

    destination.parent.mkdir(parents=True, exist_ok=True)
    staging = Path(tempfile.mkdtemp(prefix=f".{destination.name}.staging-", dir=destination.parent))
    try:
        with zipfile.ZipFile(archive_path) as outer:
            entries = [(info, safe_parts(info.filename)) for info in outer.infolist()]
            if len({parts[0] for _, parts in entries}) != 1:
                raise ValueError("ZIP harus memiliki satu folder akar")
            for info, parts in entries:
                if info.is_dir():
                    continue
                nested_zip = parts[-1].lower().endswith(".zip")
                class_dir = Path(parts[-1]).stem if nested_zip else parts[1]
                if class_dir not in CLASS_DIR_TO_LABEL:
                    raise ValueError(f"Folder kelas tidak dikenal: {class_dir}")
                label_dir = staging / CLASS_DIR_TO_LABEL[class_dir]
                if nested_zip:
                    if len(parts) != 2:
                        raise ValueError(f"ZIP kelas tidak berada di akar: {info.filename}")
                    with tempfile.TemporaryFile(dir=staging) as nested_file:
                        with outer.open(info) as nested_source:
                            shutil.copyfileobj(nested_source, nested_file)
                        nested_file.seek(0)
                        with zipfile.ZipFile(nested_file) as nested:
                            extract_members(nested, label_dir, strip_first=class_dir)
                else:
                    if len(parts) < 3:
                        raise ValueError(f"Path gambar tidak lengkap: {info.filename}")
                    write_member(
                        outer, info, staging,
                        (CLASS_DIR_TO_LABEL[class_dir], *parts[2:]),
                    )
        os.replace(staging, destination)
    except BaseException:
        shutil.rmtree(staging, ignore_errors=True)
        raise


def sha256_file(path):
    digest = hashlib.sha256()
    with Path(path).open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def difference_hash64(image):
    pixels = list(image.convert("L").resize((9, 8), Image.Resampling.LANCZOS).getdata())
    return sum(
        1 << bit for bit in range(64)
        if pixels[(bit // 8) * 9 + bit % 8] > pixels[(bit // 8) * 9 + bit % 8 + 1]
    )

def exact_duplicate_decisions(image_records):
    valid = [dict(row) for row in image_records if row["status"] == "valid"]
    by_hash = defaultdict(list)
    for row in valid:
        by_hash[row["sha256"]].append(row)
    unique, decisions = [], {}
    for group in by_hash.values():
        ordered = sorted(group, key=lambda row: row["relative_path"].casefold())
        labels = {row["class_name"] for row in ordered}
        if len(labels) > 1:
            for row in ordered:
                decisions[row["relative_path"]] = ("ambiguous_exact_duplicate", "")
        else:
            unique.append(ordered[0])
            for row in ordered[1:]:
                decisions[row["relative_path"]] = (
                    "exact_duplicate_excluded", ordered[0]["relative_path"]
                )
    return unique, decisions


def find_near_duplicate_pairs(image_records, threshold=NEAR_DUPLICATE_HAMMING_THRESHOLD):
    unique, _ = exact_duplicate_decisions(image_records)
    hashed = [(row, int(row["perceptual_hash"], 16)) for row in unique if row["perceptual_hash"]]
    # ponytail: O(N^2) for this 7.9k-image set; use indexed Hamming search if it grows.
    pairs = []
    for index, (left, left_hash) in enumerate(hashed):
        for right, right_hash in hashed[index + 1:]:
            distance = (left_hash ^ right_hash).bit_count()
            if distance <= threshold:
                pairs.append({
                    "path_left": left["relative_path"], "path_right": right["relative_path"],
                    "class_left": left["class_name"], "class_right": right["class_name"],
                    "hamming_distance": distance,
                })
    return pairs

def connected_components(image_records, pairs):
    by_path = {row["relative_path"]: row for row in image_records}
    parent = {path: path for path in by_path}

    def find(path):
        while parent[path] != path:
            parent[path] = parent[parent[path]]
            path = parent[path]
        return path

    for pair in pairs:
        left, right = pair["path_left"], pair["path_right"]
        if left in parent and right in parent:
            a, b = find(left), find(right)
            if a != b:
                first, second = sorted((a, b))
                parent[second] = first
    grouped = defaultdict(list)
    for path, row in by_path.items():
        grouped[find(path)].append(row)
    components, group_ids = list(grouped.values()), {}
    for group in components:
        paths = sorted(row["relative_path"] for row in group)
        group_id = hashlib.sha256("\n".join(paths).encode()).hexdigest()[:16]
        for path in paths:
            group_ids[path] = group_id
    return components, group_ids


def target_counts(count):
    raw = {split: count * ratio for split, ratio in SPLIT_RATIOS.items()}
    targets = {split: int(value) for split, value in raw.items()}
    remainder = count - sum(targets.values())
    order = sorted(SPLITS, key=lambda split: (-(raw[split] - targets[split]), SPLITS.index(split)))
    for split in order[:remainder]:
        targets[split] += 1
    return targets


def assign_groups(components, seed):
    totals = Counter(row["class_name"] for group in components for row in group)
    targets = {label: target_counts(totals[label]) for label in CLASS_LABELS}
    current = {label: {split: 0 for split in SPLITS} for label in CLASS_LABELS}
    stable = []
    for group in components:
        paths = sorted(row["relative_path"] for row in group)
        key = hashlib.sha256("\n".join(paths).encode()).hexdigest()
        stable.append((group, key))
    stable.sort(key=lambda item: (-len(item[0]), hashlib.sha256(f"{seed}:{item[1]}".encode()).hexdigest()))
    tie_order = sorted(SPLITS, key=lambda split: hashlib.sha256(f"{seed}:{split}".encode()).hexdigest())
    assignments = {}
    for group, _ in stable:
        group_counts = Counter(row["class_name"] for row in group)
        costs = {}
        for split in SPLITS:
            delta = 0.0
            for label, amount in group_counts.items():
                target = max(targets[label][split], 1)
                before = current[label][split] - targets[label][split]
                after = before + amount
                delta += (after * after - before * before) / (target * target)
            costs[split] = delta
        chosen = min(SPLITS, key=lambda split: (costs[split], tie_order.index(split)))
        for row in group:
            path = row["relative_path"]
            assignments[path] = chosen
            current[row["class_name"]][chosen] += 1
    return assignments

def write_csv(path, rows, fields):
    with Path(path).open("w", newline="", encoding="utf-8") as stream:
        writer = csv.DictWriter(stream, fieldnames=fields, extrasaction="ignore")
        writer.writeheader()
        writer.writerows(rows)

def prepare_dataset():
    import matplotlib.pyplot as plt

    if not SOURCE_ARCHIVE.is_file():
        raise FileNotFoundError(SOURCE_ARCHIVE)
    existing = [
        PROCESSED_DIR / name
        for name in (
            "audit.json",
            "split_manifest.csv",
            "exact_duplicates.csv",
            "near_duplicate_candidates.csv",
            "class_distribution.png",
        )
    ]
    if any(path.exists() for path in existing):
        raise FileExistsError(
            "Artefak persiapan sudah ada. Script tidak mengaudit atau menimpa ulang; "
            "gunakan data tersebut untuk training."
        )
    PROCESSED_DIR.mkdir(parents=True, exist_ok=True)
    SOURCE_ARCHIVE_SHA256 = sha256_file(SOURCE_ARCHIVE)


    if not EXTRACTED_DIR.exists():
        safe_extract_dataset(SOURCE_ARCHIVE, EXTRACTED_DIR)
    else:
        print("Ekstraksi sudah tersedia; folder tidak ditimpa:", EXTRACTED_DIR)

    records = []
    image_paths = sorted(EXTRACTED_DIR.rglob("*"), key=lambda item: item.as_posix().casefold())
    for image_index, path in enumerate(image_paths, start=1):
        if path.is_symlink() or not path.is_file():
            continue
        relative = path.relative_to(EXTRACTED_DIR).as_posix()
        parts = PurePosixPath(relative).parts
        label = parts[0] if parts and parts[0] in CLASS_LABELS else "unknown_class"
        row = {
            "relative_path": relative, "class_name": label, "status": "corrupt",
            "sha256": sha256_file(path), "perceptual_hash": "", "extension": path.suffix.lower(),
            "format": "", "width": "", "height": "", "mode": "",
            "file_size_bytes": path.stat().st_size, "error": "",
        }
        try:
            with Image.open(path) as image:
                image.verify()
            with Image.open(path) as image:
                image.load()
                row.update({
                    "format": image.format or "", "width": image.width, "height": image.height,
                    "mode": image.mode, "perceptual_hash": f"{difference_hash64(image):016x}",
                })
            row["status"] = (
                "unknown_class" if label == "unknown_class"
                else "valid" if row["format"] in {"JPEG", "PNG"}
                else "unsupported_format"
            )
        except Exception as error:
            row["error"] = f"{type(error).__name__}: {error}"[:300]
        records.append(row)
        if image_index % 1000 == 0 or image_index == len(image_paths):
            print(f"Audit citra: {image_index}/{len(image_paths)}", flush=True)

    status_counts = Counter(row["status"] for row in records)
    raw_class_counts = Counter(row["class_name"] for row in records)
    valid_records = [row for row in records if row["status"] == "valid"]
    valid_class_counts = Counter(row["class_name"] for row in valid_records)
    print("Hash ZIP sumber:", SOURCE_ARCHIVE_SHA256)
    print("Jumlah citra:", len(records), "Status:", dict(status_counts))




    fig, ax = plt.subplots(figsize=(9, 4.5))
    counts = [valid_class_counts[label] for label in CLASS_LABELS]
    ax.bar(CLASS_LABELS, counts, color="#4c78a8")
    ax.set_title("Jumlah citra valid per kelas sebelum deduplikasi")
    ax.set_ylabel("Jumlah citra")
    ax.tick_params(axis="x", rotation=25)
    for tick in ax.get_xticklabels():
        tick.set_ha("right")
    fig.tight_layout()
    fig.savefig(PROCESSED_DIR / "class_distribution.png", dpi=160)
    plt.close(fig)
    print("Citra valid per kelas:", dict(valid_class_counts))
    print("Total aktual:", sum(counts), "| Total pada laporan Progress I: 8.814")




    unique_records, duplicate_decisions = exact_duplicate_decisions(records)
    near_duplicate_pairs = find_near_duplicate_pairs(records)
    cross_class_near_duplicate_pairs = [
        pair for pair in near_duplicate_pairs
        if pair["class_left"] != pair["class_right"]
    ]
    cross_class_candidate_images = {
        path for pair in cross_class_near_duplicate_pairs
        for path in (pair["path_left"], pair["path_right"])
    }
    print("Sampel unik:", len(unique_records))
    print("Duplikat persis yang dikecualikan:", sum(v[0] == "exact_duplicate_excluded" for v in duplicate_decisions.values()))
    print("Pasangan kandidat near-duplicate:", len(near_duplicate_pairs))
    print("Pasangan near-duplicate lintas kelas:", len(cross_class_near_duplicate_pairs))




    components, near_duplicate_group_ids = connected_components(unique_records, near_duplicate_pairs)
    manifest_rows = []
    for seed in DEFAULT_SEEDS:
        assignments = assign_groups(components, seed)
        for original in records:
            row = dict(original)
            path = row["relative_path"]
            decision = duplicate_decisions.get(path)
            if decision:
                row["status"], row["duplicate_of"] = decision
                row["split"], row["near_duplicate_group"] = "excluded", ""
            elif row["status"] == "valid":
                row["status"] = "included"
                row["split"] = assignments[path]
                row["near_duplicate_group"] = near_duplicate_group_ids[path]
                row["duplicate_of"] = ""
            else:
                row["split"], row["near_duplicate_group"], row["duplicate_of"] = "excluded", "", ""
            row["seed"] = seed
            manifest_rows.append(row)
    print("Baris manifest:", len(manifest_rows))




    write_csv(PROCESSED_DIR / "split_manifest.csv", manifest_rows, MANIFEST_FIELDS)
    first_seed = min(DEFAULT_SEEDS)
    write_csv(
        PROCESSED_DIR / "exact_duplicates.csv",
        [row for row in manifest_rows if row["seed"] == first_seed and row["status"] in {
            "exact_duplicate_excluded", "ambiguous_exact_duplicate"
        }],
        ("relative_path", "class_name", "sha256", "status", "duplicate_of"),
    )
    write_csv(
        PROCESSED_DIR / "near_duplicate_candidates.csv", near_duplicate_pairs,
        ("path_left", "path_right", "class_left", "class_right", "hamming_distance"),
    )
    formats = Counter(row["format"] for row in records if row["format"])
    modes = Counter(row["mode"] for row in records if row["mode"])
    resolutions = Counter(
        f"{row['width']}x{row['height']}" for row in records if row["width"] and row["height"]
    )
    unique_counts = Counter(row["class_name"] for row in unique_records)
    group_sizes = Counter(near_duplicate_group_ids.values())
    splits_by_seed = {}
    for seed in DEFAULT_SEEDS:
        selected = [row for row in manifest_rows if row["seed"] == seed]
        splits_by_seed[str(seed)] = {
            split: {
                label: sum(
                    row["status"] == "included" and row["split"] == split and row["class_name"] == label
                    for row in selected
                )
                for label in CLASS_LABELS
            }
            for split in SPLITS
        }

    audit_summary = {
        "source_archive": SOURCE_ARCHIVE.name,
        "source_archive_sha256": SOURCE_ARCHIVE_SHA256,
        "configured_seeds": list(DEFAULT_SEEDS),
        "split_ratios": SPLIT_RATIOS,
        "image_size": list(IMAGE_SIZE),
        "raw_image_files": len(records),
        "raw_count_by_class": {label: raw_class_counts[label] for label in CLASS_LABELS},
        "valid_count_by_class": {label: valid_class_counts[label] for label in CLASS_LABELS},
        "exact_unique_count_by_class": {label: unique_counts[label] for label in CLASS_LABELS},
        "file_status_counts": dict(status_counts),
        "formats": dict(formats),
        "modes": dict(modes),
        "distinct_resolutions": len(resolutions),
        "resolution_counts": dict(sorted(resolutions.items())),
        "near_duplicate_method": {
            "hash": "64-bit difference hash on 9x8 grayscale",
            "candidate_hamming_distance_max": NEAR_DUPLICATE_HAMMING_THRESHOLD,
            "candidate_pairs": len(near_duplicate_pairs),
            "cross_class_candidate_pairs": len(cross_class_near_duplicate_pairs),
            "cross_class_candidate_images": len(cross_class_candidate_images),
            "candidate_groups_ge_2": sum(size > 1 for size in group_sizes.values()),
            "images_in_candidate_groups": sum(size for size in group_sizes.values() if size > 1),
            "largest_candidate_group": max(group_sizes.values(), default=1),
            "note": "Heuristik, bukan bukti identitas daun yang sama.",
        },
        "splits_by_seed": splits_by_seed,
    }
    (PROCESSED_DIR / "audit.json").write_text(
        json.dumps(audit_summary, ensure_ascii=False, indent=2) + "\n", encoding="utf-8"
    )
    print("Sampel unik:", sum(unique_counts.values()))
    print("Manifest, audit, dan daftar kandidat tersimpan di:", PROCESSED_DIR)




    for seed in DEFAULT_SEEDS:
        selected = [
            row for row in manifest_rows if row["seed"] == seed and row["status"] == "included"
        ]
        split_totals = Counter(row["split"] for row in selected)
        print(f"Seed {seed}: {dict(split_totals)}")
        for label in CLASS_LABELS:
            class_splits = Counter(row["split"] for row in selected if row["class_name"] == label)
            assert all(class_splits[split] > 0 for split in SPLITS), (seed, label, class_splits)
        for field in ("sha256", "near_duplicate_group"):
            group_splits = defaultdict(set)
            for row in selected:
                group_splits[row[field]].add(row["split"])
            assert all(len(value) == 1 for value in group_splits.values()), field
    print("Tidak ada hash identik atau kelompok kandidat yang lintas subset.")




if __name__ == "__main__":
    prepare_dataset()
