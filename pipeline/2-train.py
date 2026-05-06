"""
Fully config-driven YOLOv8 training pipeline (production-grade)

- Uses ALL config.yaml fields
- No hardcoded training hyperparameters
- Structured logging
- Clean experiment management
"""

from __future__ import annotations

import argparse
import logging
from datetime import datetime
from pathlib import Path
from typing import Any

import yaml
from ultralytics import YOLO


# ----------------------------
# Logging
# ----------------------------
def setup_logger() -> logging.Logger:
    logger = logging.getLogger("yolo-trainer")
    logger.setLevel(logging.INFO)

    if not logger.handlers:
        handler = logging.StreamHandler()
        formatter = logging.Formatter(
            "[%(asctime)s] [%(levelname)s] %(message)s",
            datefmt="%Y-%m-%d %H:%M:%S",
        )
        handler.setFormatter(formatter)
        logger.addHandler(handler)

    return logger


logger = setup_logger()


# ----------------------------
# Config Loader
# ----------------------------
class Config:
    def __init__(self, path: Path):
        self.path = path
        self.raw = self._load()

        self.training = self.raw.get("training", {})
        self.preprocess = self.raw.get("preprocess", {})
        self.validation = self.raw.get("validation", {})
        self.deployment = self.raw.get("deployment", {})

        self._validate()

    def _load(self) -> dict[str, Any]:
        if not self.path.exists():
            raise FileNotFoundError(f"Config not found: {self.path}")

        with open(self.path, "r") as f:
            return yaml.safe_load(f)

    def _validate(self):
        required = ["model_name", "data_yaml"]
        for r in required:
            if r not in self.training:
                raise ValueError(f"Missing required training key: {r}")


# ----------------------------
# Trainer
# ----------------------------
class YOLOTrainer:
    def __init__(self, config: Config):
        self.config = config

        self.run_name = self._make_run_name()
        self.project = self.config.training.get("project", "runs/detect")
        self.output_dir = Path(self.project) / self.run_name

    def _make_run_name(self) -> str:
        base = self.config.training.get("name", "experiment")
        ts = datetime.now().strftime("%Y%m%d_%H%M%S")
        return f"{base}_{ts}"

    def _validate_data(self):
        data_yaml = Path(self.config.training["data_yaml"])

        if not data_yaml.exists():
            raise FileNotFoundError(f"data.yaml not found: {data_yaml}")

        logger.info("Dataset config validated")

    def train(self) -> Path:
        logger.info("🚀 Starting YOLOv8 training pipeline")

        t = self.config.training

        self._validate_data()

        # reproducibility seed
        seed = self.config.preprocess.get("seed", 42)

        model = YOLO(f"{t['model_name']}.pt")

        logger.info(f"Model: {t['model_name']}")
        logger.info(f"Run: {self.run_name}")

        results = model.train(
            # dataset
            data=t["data_yaml"],

            # core training
            epochs=t.get("epochs", 100),
            batch=t.get("batch_size", 8),
            device=t.get("device", "cpu"),
            patience=t.get("patience", 20),
            imgsz=t.get("imgsz", 640),

            # optimization
            lr0=t.get("learning_rate", 0.001),
            momentum=t.get("momentum", 0.937),
            weight_decay=t.get("weight_decay", 0.0005),

            # performance
            workers=t.get("workers", 2),
            half=t.get("half", False),
            seed=seed,

            # augmentation (FULL SUPPORT)
            augment=t.get("augment", True),
            mosaic=t.get("mosaic", 1.0),
            flipud=t.get("flipud", 0.5),
            fliplr=t.get("fliplr", 0.5),
            degrees=t.get("degrees", 10),
            translate=t.get("translate", 0.1),
            scale=t.get("scale", 0.5),

            # output
            project=self.project,
            name=self.run_name,
            save=True,
            save_period=t.get("save_period", 10),
            verbose=True,
        )

        # Use the actual save_dir from YOLO results instead of calculating it
        best_model = Path(results.save_dir) / "weights" / "best.pt"

        if not best_model.exists():
            raise RuntimeError(f"Training finished but best.pt not found at {best_model}")

        logger.info(f"✅ Training complete: {best_model}")

        return best_model

    def validate(self):
        """Optional validation hook (uses validation config)"""
        v = self.config.validation

        if not v:
            logger.info("No validation config provided, skipping.")
            return

        logger.info("🔍 Validation config detected (placeholder hook)")

        # You can extend this with YOLO.val() if needed


    def get_deployment_model(self) -> Path | None:
        """Return deployment model path from config"""
        path = self.config.deployment.get("model_path")
        if path:
            return Path(path)
        return None


# ----------------------------
# CLI
# ----------------------------
def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", default="config.yaml")
    args = parser.parse_args()

    config = Config(Path(args.config))
    trainer = YOLOTrainer(config)

    best_model = trainer.train()

    logger.info(f"🎯 Best model ready at: {best_model}")

    deploy_model = trainer.get_deployment_model()
    if deploy_model:
        logger.info(f"📦 Deployment model (from config): {deploy_model}")


if __name__ == "__main__":
    main()