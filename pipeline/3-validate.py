"""
Validation script for trained YOLOv8n car detection model.

Purpose: Evaluate trained model on test set

Responsibilities:
- Load trained model
- Run inference on test images
- Compute evaluation metrics (mAP, precision, recall, F1)
- Generate metrics report
- Visualize detections on test images

Execution:
    python validate.py
    
Output:
- validation_results/metrics.json - Quantitative results
- validation_results/images/ - Annotated test images
"""

import json
import logging
import sys
from pathlib import Path

import cv2
import yaml
from ultralytics import YOLO

# Setup logging
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(name)s - %(levelname)s - %(message)s",
)
logger = logging.getLogger(__name__)


def load_config(config_path: str = "config.yaml") -> dict:
    """Load configuration from YAML file."""
    logger.info(f"Loading configuration from {config_path}")
    with open(config_path) as f:
        config = yaml.safe_load(f)
    return config


def find_best_model(runs_dir: str = "runs/detect") -> str:
    """
    Find the best model from training runs.
    
    Returns the path to the most recent best.pt model.
    """
    runs_path = Path(runs_dir)
    if not runs_path.exists():
        raise FileNotFoundError(f"Runs directory not found: {runs_dir}")

    # Find all best.pt files (handles nested directory structures)
    best_models = sorted(runs_path.glob("**/weights/best.pt"), key=lambda p: p.parent.parent.stat().st_mtime, reverse=True)
    
    if not best_models:
        raise FileNotFoundError(f"No trained models found in {runs_dir}")

    best_model_path = best_models[0]
    logger.info(f"Found best model: {best_model_path}")
    return str(best_model_path)


def load_test_images(data_yaml: str) -> list[str]:
    """Load test image paths from data.yaml."""
    with open(data_yaml) as f:
        data_config = yaml.safe_load(f)

    test_dir = Path(data_config.get("test", "data/images/test"))
    
    if not test_dir.exists():
        raise FileNotFoundError(f"Test directory not found: {test_dir}")

    # Get all image files
    image_extensions = {".jpg", ".jpeg", ".png", ".bmp", ".gif"}
    image_paths = [
        str(p) for p in sorted(test_dir.rglob("*"))
        if p.suffix.lower() in image_extensions
    ]

    if not image_paths:
        logger.warning(f"No test images found in {test_dir}")

    return image_paths


def run_validation(model: YOLO, data_yaml: str, config: dict) -> dict:
    """
    Run validation on test dataset.
    
    Args:
        model: Loaded YOLO model
        data_yaml: Path to data.yaml
        config: Configuration dictionary
        
    Returns:
        Dictionary with validation metrics
    """
    validation_cfg = config.get("validation", {})
    
    logger.info("=" * 70)
    logger.info("MODEL VALIDATION")
    logger.info("=" * 70)

    # Using YOLO's built-in validation
    logger.info(f"Running validation on dataset: {data_yaml}")
    
    val_results = model.val(
        data=data_yaml,
        conf=validation_cfg.get("confidence_threshold", 0.5),
        iou=validation_cfg.get("iou_threshold", 0.45),
        device=validation_cfg.get("device", "cpu"),
        verbose=True,
    )

    # Extract metrics
    metrics = {
        "model_path": str(model.ckpt_path) if hasattr(model, "ckpt_path") else "unknown",
        "box": {
            "precision": float(val_results.box.mp) if hasattr(val_results, "box") and hasattr(val_results.box, "mp") else None,
            "recall": float(val_results.box.mr) if hasattr(val_results, "box") and hasattr(val_results.box, "mr") else None,
            "mAP50": float(val_results.box.map50) if hasattr(val_results, "box") and hasattr(val_results.box, "map50") else None,
            "mAP50-95": float(val_results.box.map) if hasattr(val_results, "box") and hasattr(val_results.box, "map") else None,
        },
        "speed": {
            "preprocess": float(val_results.speed["preprocess"]) if isinstance(val_results.speed, dict) else None,
            "inference": float(val_results.speed["inference"]) if isinstance(val_results.speed, dict) else None,
            "postprocess": float(val_results.speed["postprocess"]) if isinstance(val_results.speed, dict) else None,
        }
    }

    logger.info("=" * 70)
    logger.info("VALIDATION METRICS")
    logger.info("=" * 70)
    for key, value in metrics.items():
        if isinstance(value, dict):
            logger.info(f"{key}:")
            for k, v in value.items():
                logger.info(f"  {k}: {v}")
        else:
            logger.info(f"{key}: {value}")

    return metrics


def annotate_images(
    model: YOLO,
    image_paths: list[str],
    output_dir: str,
    config: dict,
) -> None:
    """
    Annotate test images with model predictions.
    
    Args:
        model: Loaded YOLO model
        image_paths: List of image file paths
        output_dir: Output directory for annotated images
        config: Configuration dictionary
    """
    validation_cfg = config.get("validation", {})
    confidence = validation_cfg.get("confidence_threshold", 0.5)

    output_path = Path(output_dir) / "images"
    output_path.mkdir(parents=True, exist_ok=True)

    logger.info(f"Annotating {len(image_paths)} test images...")

    for idx, image_file in enumerate(image_paths, 1):
        image_path = Path(image_file)
        
        # Read image
        image = cv2.imread(image_file)
        if image is None:
            logger.warning(f"Could not read image: {image_file}")
            continue

        # Run inference
        results = model.predict(image, conf=confidence, verbose=False)

        # Annotate
        annotated = results[0].plot()

        # Save
        output_file = output_path / image_path.name
        cv2.imwrite(str(output_file), annotated)
        
        if idx % max(1, len(image_paths) // 10) == 0:
            logger.info(f"  Annotated {idx}/{len(image_paths)} images")

    logger.info(f"✓ Annotated images saved to {output_path}")


def main():
    """Main validation pipeline."""
    try:
        # Load config
        config = load_config("config.yaml")
        validation_cfg = config.get("validation", {})

        if not validation_cfg:
            logger.warning("No validation configuration found in config.yaml")
            logger.info("Skipping validation.")
            return

        # Setup output directory
        results_dir = validation_cfg.get("results_dir", "validation_results")
        results_path = Path(results_dir)
        results_path.mkdir(parents=True, exist_ok=True)

        # Find best model
        logger.info("Looking for trained model...")
        model_path = find_best_model()
        logger.info(f"Loading model from: {model_path}")
        model = YOLO(model_path)

        # Get test data
        data_yaml = config.get("training", {}).get("data_yaml", "data/data.yaml")
        test_images = load_test_images(data_yaml)

        if not test_images:
            logger.warning("No test images found. Skipping annotation.")
        else:
            # Run validation
            metrics = run_validation(model, data_yaml, config)

            # Save metrics
            metrics_file = results_path / "metrics.json"
            with open(metrics_file, "w") as f:
                json.dump(metrics, f, indent=2)
            logger.info(f"✓ Metrics saved to {metrics_file}")

            # Annotate images
            if validation_cfg.get("save_results", True):
                annotate_images(model, test_images, results_dir, config)

        logger.info("=" * 70)
        logger.info("✓ VALIDATION COMPLETE")
        logger.info("=" * 70)
        logger.info(f"Results directory: {results_dir}")
        logger.info("=" * 70)

    except Exception as e:
        logger.error(f"Validation failed: {str(e)}")
        sys.exit(1)


if __name__ == "__main__":
    main()
