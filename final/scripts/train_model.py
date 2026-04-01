"""Train / fine-tune a YOLO model on the snooker dataset.

Usage:
  # Full fine-tune (need complete labels):
  python final/scripts/train_model.py --base-model best_color.pt --epochs 50

  # Freeze backbone, only train detection head (safe for partial labels):
  python final/scripts/train_model.py --base-model best_color.pt --epochs 50 --freeze

After training, copy the best model:
  cp runs/detect/train/weights/best.pt final/src/snookervision/data/model/best_color.pt
"""
import argparse
import shutil
import os


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--dataset", default="final/scripts/dataset/data.yaml",
                        help="Path to data.yaml")
    parser.add_argument("--epochs", type=int, default=100)
    parser.add_argument("--imgsz", type=int, default=640)
    parser.add_argument("--batch", type=int, default=16)
    parser.add_argument("--base-model",
                        default="final/src/snookervision/data/model/best_color.pt",
                        help="Pretrained base model to fine-tune from")
    parser.add_argument("--device", default="auto",
                        help="Training device: auto, cpu, cuda, mps")
    parser.add_argument("--freeze", type=int, nargs="?", const=10, default=0,
                        help="Freeze first N layers (default 10 = backbone). "
                             "Use --freeze for safe fine-tune with partial labels.")
    parser.add_argument("--lr", type=float, default=0.001,
                        help="Learning rate (default 0.001, use 0.0005 for gentle fine-tune)")
    parser.add_argument("--install", action="store_true",
                        help="Auto-copy best.pt to the model directory after training")
    args = parser.parse_args()

    if not os.path.exists(args.dataset):
        print(f"Dataset not found at {args.dataset}")
        print("Run these first:")
        print("  python final/scripts/collect_frames.py --source <camera_url>")
        print("  python final/scripts/auto_label.py")
        return

    from ultralytics import YOLO
    import torch

    device = args.device
    if device == "auto":
        if torch.cuda.is_available():
            device = "cuda"
        elif hasattr(torch.backends, "mps") and torch.backends.mps.is_available():
            device = "mps"
        else:
            device = "cpu"

    print(f"Training on device: {device}")
    print(f"Dataset: {args.dataset}")
    print(f"Base model: {args.base_model}")
    print(f"Epochs: {args.epochs}, Image size: {args.imgsz}, Batch: {args.batch}")
    if args.freeze:
        print(f"Freezing first {args.freeze} layers (backbone)")
    print(f"Learning rate: {args.lr}")
    print()

    model = YOLO(args.base_model)
    results = model.train(
        data=os.path.abspath(args.dataset),
        epochs=args.epochs,
        imgsz=args.imgsz,
        batch=args.batch,
        device=device,
        patience=20,
        save=True,
        plots=True,
        verbose=True,
        freeze=args.freeze,
        lr0=args.lr,
    )

    # Find best.pt
    best_pt = None
    for candidate in [
        "runs/detect/train/weights/best.pt",
        "runs/detect/train2/weights/best.pt",
        "runs/detect/train3/weights/best.pt",
        "runs/detect/train4/weights/best.pt",
        "runs/detect/train5/weights/best.pt",
    ]:
        if os.path.exists(candidate):
            best_pt = candidate

    if best_pt:
        print(f"\nTraining complete! Best model: {best_pt}")

        if args.install:
            dest = "final/src/snookervision/data/model/best_color.pt"
            shutil.copy2(best_pt, dest)
            print(f"Installed to {dest}")
        else:
            print(f"\nTo use the new model, run:")
            print(f"  cp {best_pt} final/src/snookervision/data/model/best_color.pt")
    else:
        print("\nTraining complete! Check runs/detect/ for the best.pt file.")


if __name__ == "__main__":
    main()
