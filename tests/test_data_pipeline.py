import ast
import csv
import hashlib
import importlib.util
import io
import tempfile
import unittest
import zipfile
from collections import Counter, defaultdict
from pathlib import Path

import nbformat


ROOT = Path(__file__).resolve().parents[1]
NOTEBOOK_PATH = ROOT / "Klasifikasi_Daun_Cabai.ipynb"
PREP_PATH = ROOT / "scripts" / "prepare_data.py"


def load_prep_module():
    spec = importlib.util.spec_from_file_location("prepare_data", PREP_PATH)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


class NotebookPipelineTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.prep = load_prep_module()

    def test_notebook_and_preparation_script_parse(self):
        notebook = nbformat.read(NOTEBOOK_PATH, as_version=4)
        nbformat.validate(notebook)
        code_sources = []
        for index, cell in enumerate(notebook.cells):
            if cell.cell_type == "code":
                ast.parse(cell.source, filename=f"cell-{index}")
                code_sources.append(cell.source)
        ast.parse(PREP_PATH.read_text(encoding="utf-8"))

        combined = "\n".join(code_sources)
        self.assertNotIn("safe_extract_dataset", combined)
        self.assertNotIn("find_near_duplicate_pairs", combined)
        self.assertIn("JALANKAN_BASELINE = False", combined)

    def test_random_augmentation_is_only_selected_for_training_rows(self):
        notebook = nbformat.read(NOTEBOOK_PATH, as_version=4)
        transform_source = next(cell.source for cell in notebook.cells if "def make_transform" in cell.source)
        tree = ast.parse(transform_source)
        function = next(
            node for node in tree.body
            if isinstance(node, ast.FunctionDef) and node.name == "make_transform"
        )
        training_branch = next(
            node for node in ast.walk(function)
            if isinstance(node, ast.If) and ast.unparse(node.test) == "training"
        )
        transforms = {
            node.func.attr for node in ast.walk(training_branch)
            if isinstance(node, ast.Call) and isinstance(node.func, ast.Attribute)
        }
        self.assertTrue(
            {"RandomHorizontalFlip", "RandomRotation", "ColorJitter"}.issubset(transforms)
        )
        loader_source = next(
            cell.source for cell in notebook.cells if "def build_loaders" in cell.source
        )
        self.assertIn("training=(split == \"train\")", loader_source)

    def test_nested_extraction_preserves_archive_and_rejects_unsafe_paths(self):
        with tempfile.TemporaryDirectory() as temp_name:
            root = Path(temp_name)
            nested_bytes = io.BytesIO()
            with zipfile.ZipFile(nested_bytes, "w") as archive:
                archive.writestr("Bacterial_Spot/one.png", b"nested fixture")

            source = root / "dataset.zip"
            with zipfile.ZipFile(source, "w") as archive:
                archive.writestr("dataset/Bacterial_Spot.zip", nested_bytes.getvalue())
                archive.writestr("dataset/Powdery_Mildew/two.png", b"direct fixture")
            archive_hash = hashlib.sha256(source.read_bytes()).hexdigest()

            destination = root / "extracted"
            self.prep.safe_extract_dataset(source, destination)
            self.assertEqual(
                (destination / "Bacterial Spot" / "one.png").read_bytes(),
                b"nested fixture",
            )
            self.assertEqual(
                (destination / "Powdery Mildew" / "two.png").read_bytes(),
                b"direct fixture",
            )
            self.assertEqual(hashlib.sha256(source.read_bytes()).hexdigest(), archive_hash)
            with self.assertRaises(FileExistsError):
                self.prep.safe_extract_dataset(source, destination)

            for unsafe in ("../escape.png", "C:/escape.png", "folder:name.png", r"a\\b.png", "bad\x00name"):
                with self.subTest(path=unsafe):
                    with self.assertRaises(ValueError):
                        self.prep.safe_parts(unsafe)

            malicious = root / "malicious.zip"
            with zipfile.ZipFile(malicious, "w") as archive:
                archive.writestr("dataset/Bacterial_Spot/../escape.png", b"bad")
            rejected = root / "rejected"
            with self.assertRaises(ValueError):
                self.prep.safe_extract_dataset(malicious, rejected)
            self.assertFalse(rejected.exists())
            self.assertFalse((root / "escape.png").exists())

    def test_group_split_is_deterministic_and_keeps_candidate_groups_together(self):
        records = [
            {
                "relative_path": f"{label}/{index:02}.png",
                "class_name": label,
            }
            for label in self.prep.CLASS_LABELS
            for index in range(10)
        ]
        pair_paths = (records[0]["relative_path"], records[1]["relative_path"])
        pairs = [{"path_left": pair_paths[0], "path_right": pair_paths[1]}]
        components, group_ids = self.prep.connected_components(records, pairs)

        first = self.prep.assign_groups(components, seed=42)
        second = self.prep.assign_groups(components, seed=42)

        self.assertEqual(first, second)
        self.assertEqual(first[pair_paths[0]], first[pair_paths[1]])
        self.assertEqual(group_ids[pair_paths[0]], group_ids[pair_paths[1]])
        for label in self.prep.CLASS_LABELS:
            class_splits = {
                first[row["relative_path"]]
                for row in records
                if row["class_name"] == label
            }
            self.assertEqual(class_splits, set(self.prep.SPLITS))

    def test_existing_manifest_matches_group_split_for_each_seed(self):
        manifest_path = ROOT / "data" / "processed" / "split_manifest.csv"
        candidates_path = ROOT / "data" / "processed" / "near_duplicate_candidates.csv"
        self.assertTrue(manifest_path.is_file())
        self.assertTrue(candidates_path.is_file())
        with manifest_path.open(newline="", encoding="utf-8-sig") as stream:
            rows = list(csv.DictReader(stream))
        with candidates_path.open(newline="", encoding="utf-8-sig") as stream:
            pairs = list(csv.DictReader(stream))

        for seed in self.prep.DEFAULT_SEEDS:
            selected = [
                row for row in rows
                if int(row["seed"]) == seed and row["status"] == "included"
            ]
            components, group_ids = self.prep.connected_components(selected, pairs)
            expected = self.prep.assign_groups(components, seed)
            actual = {row["relative_path"]: row["split"] for row in selected}
            self.assertEqual(actual, expected)
            self.assertEqual(
                {row["relative_path"]: row["near_duplicate_group"] for row in selected},
                group_ids,
            )

            for field in ("sha256", "near_duplicate_group"):
                split_by_group = defaultdict(set)
                for row in selected:
                    split_by_group[row[field]].add(row["split"])
                self.assertTrue(all(len(splits) == 1 for splits in split_by_group.values()))

            class_split_counts = Counter(
                (row["class_name"], row["split"]) for row in selected
            )
            self.assertEqual(
                set(class_split_counts),
                {(label, split) for label in self.prep.CLASS_LABELS for split in self.prep.SPLITS},
            )


if __name__ == "__main__":
    unittest.main()
