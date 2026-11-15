import ast
import json
import tempfile
import unittest
from pathlib import Path

import torch
from torch.utils.data import DataLoader, TensorDataset


ROOT = Path(__file__).resolve().parents[1]
NOTEBOOK_PATH = ROOT / "Klasifikasi_Daun_Cabai.ipynb"
CLASS_LABELS = (
    "Bacterial Spot",
    "Cercospora Leaf Spot",
    "Curl Virus",
    "Healthy Leaf",
    "Nutrient Deficiency",
    "Powdery Mildew",
)


def load_notebook_functions(names):
    notebook = json.loads(NOTEBOOK_PATH.read_text(encoding="utf-8"))
    definitions = []
    for cell_index, cell in enumerate(notebook["cells"]):
        if cell["cell_type"] != "code":
            continue
        tree = ast.parse("".join(cell["source"]), filename=f"cell-{cell_index}")
        definitions.extend(
            node for node in tree.body
            if isinstance(node, ast.FunctionDef) and node.name in names
        )
    found = {node.name for node in definitions}
    if found != set(names):
        raise AssertionError(f"Missing notebook functions: {set(names) - found}")

    namespace = {
        "CLASS_LABELS": CLASS_LABELS,
        "CLASS_TO_INDEX": {label: index for index, label in enumerate(CLASS_LABELS)},
        "torch": torch,
        "Path": Path,
    }
    module = ast.Module(body=definitions, type_ignores=[])
    exec(compile(module, "notebook-baseline-functions", "exec"), namespace, namespace)
    return namespace


class ToyModel(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.features = torch.nn.Sequential(
            torch.nn.Flatten(),
            torch.nn.Linear(3 * 2 * 2, 8),
            torch.nn.ReLU(),
        )
        self.classifier = torch.nn.Linear(8, len(CLASS_LABELS))

    def forward(self, images):
        return self.classifier(self.features(images))


class FixedPredictionModel(torch.nn.Module):
    def forward(self, images):
        sample_ids = images[:, 0, 0, 0].long()
        predictions = torch.tensor([0, 1, 0, 1], device=images.device)[sample_ids]
        logits = torch.zeros(len(images), len(CLASS_LABELS), device=images.device)
        return logits.scatter_(1, predictions[:, None], 10)


class BaselineTrainingTests(unittest.TestCase):
    def test_class_weights_use_only_selected_seed_training_rows(self):
        functions = load_notebook_functions({"class_weights_from_manifest"})
        rows = []
        for index, label in enumerate(CLASS_LABELS, start=1):
            for seed, split, count in (
                (42, "train", index),
                (42, "validation", 25),
                (42, "test", 30),
                (43, "train", 100),
            ):
                rows.extend(
                    {"seed": seed, "split": split, "status": "included", "class_name": label}
                    for _ in range(count)
                )

        weights = functions["class_weights_from_manifest"](rows, seed=42)
        total = sum(range(1, len(CLASS_LABELS) + 1))
        expected = torch.tensor(
            [total / (len(CLASS_LABELS) * count) for count in range(1, len(CLASS_LABELS) + 1)]
        )
        self.assertTrue(torch.allclose(weights, expected))

    def test_macro_f1_includes_all_classes_and_uses_zero_for_missing_classes(self):
        functions = load_notebook_functions({"run_epoch"})
        images = torch.zeros(4, 3, 2, 2)
        images[:, 0, 0, 0] = torch.arange(4)
        labels = torch.tensor([0, 0, 1, 1])
        loader = DataLoader(TensorDataset(images, labels), batch_size=4)

        _, macro_f1 = functions["run_epoch"](
            FixedPredictionModel(),
            loader,
            torch.nn.CrossEntropyLoss(),
            torch.device("cpu"),
        )

        self.assertAlmostEqual(macro_f1, 1 / 6, places=6)

    def test_two_stage_training_backpropagates_and_saves_validation_checkpoints(self):
        functions = load_notebook_functions({"run_epoch", "train_model"})
        model = ToyModel()
        images = torch.randn(len(CLASS_LABELS), 3, 2, 2)
        labels = torch.arange(len(CLASS_LABELS))
        loader = DataLoader(TensorDataset(images, labels), batch_size=3, shuffle=False)
        criterion = torch.nn.CrossEntropyLoss()
        config = {
            "head_epochs": 1,
            "head_learning_rate": 0.1,
            "fine_tune_epochs": 1,
            "fine_tune_learning_rate": 0.05,
            "weight_decay": 1e-4,
            "cosine_eta_min": 1e-6,
        }
        initial_features = model.features[1].weight.detach().clone()
        initial_classifier = model.classifier.weight.detach().clone()

        with tempfile.TemporaryDirectory() as temp_dir:
            history = functions["train_model"](
                model=model,
                train_loader=loader,
                validation_loader=loader,
                criterion=criterion,
                device=torch.device("cpu"),
                seed=42,
                checkpoint_dir=temp_dir,
                config=config,
            )

            self.assertEqual([row["stage"] for row in history], ["classifier_head", "fine_tune_all"])
            self.assertTrue(torch.isfinite(torch.tensor([row["train_loss"] for row in history])).all())
            self.assertTrue(
                torch.isfinite(torch.tensor([row["validation_macro_f1"] for row in history])).all()
            )
            self.assertFalse(torch.equal(initial_classifier, model.classifier.weight.detach()))
            self.assertFalse(torch.equal(initial_features, model.features[1].weight.detach()))

            for stage in ("classifier_head", "fine_tune_all"):
                checkpoint_path = Path(temp_dir) / f"{stage}_best.pth"
                self.assertTrue(checkpoint_path.is_file())
                checkpoint = torch.load(checkpoint_path, map_location="cpu", weights_only=True)
                self.assertEqual(checkpoint["architecture"], "MobileNetV3-Small")
                self.assertEqual(checkpoint["seed"], 42)
                self.assertEqual(checkpoint["stage"], stage)
                self.assertEqual(checkpoint["class_labels"], list(CLASS_LABELS))
                self.assertIn("validation_macro_f1", checkpoint)


if __name__ == "__main__":
    unittest.main()
