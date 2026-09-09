"""Train row detection. Input: rowdet_obb quads. Processing: YOLO11s-OBB fine-tuning. Output: models/row_detector/best.pt."""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from scripts._training import (
    backup_yolo_previous,
    ensure_ultralytics_mps_safe,
    find_yolo_checkpoint,
    graceful_training_interrupts,
    print_training_banner,
    runtime_yolo_yaml,
    save_yolo_best,
    training_lock,
)
from src.utils import PROJECT_ROOT, choose_device


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--data", type=Path, default=PROJECT_ROOT / "data/rowdet_obb")
    parser.add_argument("--model", default="yolo11s-obb.pt")
    parser.add_argument("--epochs", type=int, default=60)
    parser.add_argument("--imgsz", type=int, default=320)
    parser.add_argument("--batch", type=int, default=32)
    parser.add_argument("--device", default="auto")
    mode = parser.add_mutually_exclusive_group()
    mode.add_argument("--resume", action="store_true", help="Require a valid incomplete checkpoint")
    mode.add_argument("--fresh", action="store_true", help="Start a new run and preserve all prior runs")
    parser.add_argument("--workers", type=int, default=0)
    parser.add_argument("--run-name", default="row_detector_obb", help=argparse.SUPPRESS)
    parser.add_argument("--output", type=Path, default=PROJECT_ROOT / "models/row_detector/best.pt", help=argparse.SUPPRESS)
    parser.add_argument("--stop-after-epoch", type=int, default=0, help=argparse.SUPPRESS)
    args = parser.parse_args()
    from ultralytics import YOLO

    device = choose_device(args.device)
    ensure_ultralytics_mps_safe(device)
    yaml_path = runtime_yolo_yaml(args.data)
    checkpoint = None if args.fresh else find_yolo_checkpoint(
        run_name=args.run_name,
        task="obb",
        model_name=args.model,
        yaml_path=yaml_path,
        imgsz=args.imgsz,
        epochs=args.epochs,
        expected_names={0: "text_row"},
        require=args.resume,
    )
    if checkpoint and checkpoint.completed_epochs >= checkpoint.target_epochs:
        print(f"Training already completed: {checkpoint.completed_epochs}/{checkpoint.target_epochs} epochs.")
        print("Use --fresh to intentionally train again.")
        return 0
    print_training_banner("ROW DETECTOR (OBB)", "AUTO RESUME" if checkpoint else "NEW TRAINING", args.epochs, device, checkpoint)
    model = YOLO(str(checkpoint.path) if checkpoint else args.model)
    model.add_callback("on_train_epoch_end", backup_yolo_previous)
    if args.stop_after_epoch:
        def stop_after_checkpoint(trainer):
            if trainer.epoch + 1 >= args.stop_after_epoch:
                raise KeyboardInterrupt("controlled checkpoint smoke-test interruption")

        model.add_callback("on_model_save", stop_after_checkpoint)
    try:
        with training_lock(args.run_name), graceful_training_interrupts():
            if checkpoint:
                model.train(resume=True, device=device, imgsz=args.imgsz, batch=args.batch, workers=args.workers)
            else:
                model.train(
                    data=str(yaml_path), epochs=args.epochs, imgsz=args.imgsz, batch=args.batch, device=device,
                    workers=args.workers, project=str(PROJECT_ROOT / "runs"), name=args.run_name,
                )
    except KeyboardInterrupt:
        recovery = Path(getattr(getattr(model, "trainer", None), "last", checkpoint.path if checkpoint else PROJECT_ROOT / f"runs/{args.run_name}/weights/last.pt"))
        print(f"Training interrupted. Resume by running the same command. Last completed checkpoint: {recovery}")
        return 130
    print(save_yolo_best(model, args.output))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
x