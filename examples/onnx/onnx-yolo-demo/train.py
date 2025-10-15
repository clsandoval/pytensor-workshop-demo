"""
Complete training script for YOLO11n with PyTensor.

Features:
- COCO dataset support (person, cellphone classes)
- WandB logging for experiment tracking
- Checkpointing and resuming
- ONNX export after training
- JAX backend for GPU acceleration

Usage:
    PYTENSOR_FLAGS="floatX=float32,optimizer=fast_run" python train.py --epochs 100 --batch-size 8 --lr 0.01

IMPORTANT: PYTENSOR_FLAGS must be set BEFORE running this script!
"""

import argparse
import time
from pathlib import Path

import numpy as np

# WandB for experiment tracking
import wandb
from dataset import COCODataset, create_dataloader
from loss import yolo_loss

# Local imports
from model import build_yolo11n
from tqdm import tqdm

import pytensor
import pytensor.tensor as pt
from pytensor import function, shared


# Verify float32 configuration
if pytensor.config.floatX != "float32":
    raise RuntimeError(
        f"ERROR: PyTensor floatX is '{pytensor.config.floatX}' but must be 'float32' for ONNX export!\n"
        f"Set environment variable BEFORE running:\n"
        f"  export PYTENSOR_FLAGS='floatX=float32,optimizer=fast_run'\n"
        f"  python train.py ...\n"
        f"Or use: bash train.sh"
    )


def parse_args():
    """Parse command line arguments."""
    parser = argparse.ArgumentParser(description="Train YOLO11n on COCO dataset")

    # Training hyperparameters
    parser.add_argument(
        "--epochs", type=int, default=100, help="Number of training epochs"
    )
    parser.add_argument("--batch-size", type=int, default=8, help="Batch size")
    parser.add_argument("--lr", type=float, default=0.01, help="Initial learning rate")
    parser.add_argument("--momentum", type=float, default=0.9, help="SGD momentum")
    parser.add_argument("--weight-decay", type=float, default=5e-4, help="Weight decay")

    # Dataset
    parser.add_argument(
        "--data-dir", type=str, default="./data/coco", help="COCO dataset directory"
    )
    parser.add_argument(
        "--image-size", type=int, default=320, help="Input image size (square)"
    )
    parser.add_argument(
        "--num-workers", type=int, default=4, help="Number of data loading workers"
    )

    # Model
    parser.add_argument(
        "--num-classes",
        type=int,
        default=2,
        help="Number of classes (person, cellphone)",
    )

    # Checkpointing
    parser.add_argument(
        "--checkpoint-dir",
        type=str,
        default="./checkpoints",
        help="Directory to save checkpoints",
    )
    parser.add_argument(
        "--save-every", type=int, default=10, help="Save checkpoint every N epochs"
    )
    parser.add_argument(
        "--resume", type=str, default=None, help="Path to checkpoint to resume from"
    )

    # Logging
    parser.add_argument(
        "--wandb-project",
        type=str,
        default="yolo11n-pytensor",
        help="WandB project name",
    )
    parser.add_argument(
        "--wandb-entity", type=str, default=None, help="WandB entity (username or team)"
    )
    parser.add_argument("--no-wandb", action="store_true", help="Disable WandB logging")
    parser.add_argument(
        "--log-every", type=int, default=10, help="Log metrics every N batches"
    )

    # ONNX export
    parser.add_argument(
        "--export-onnx", action="store_true", help="Export model to ONNX after training"
    )

    return parser.parse_args()


class Trainer:
    """YOLO11n trainer with WandB logging."""

    def __init__(self, args):
        """Initialize trainer."""
        self.args = args
        self.device = "cuda" if self._check_gpu() else "cpu"

        print("\n" + "=" * 70)
        print("YOLO11n Training Configuration".center(70))
        print("=" * 70)
        print(f"Device: {self.device}")
        print(f"Batch size: {args.batch_size}")
        print(f"Learning rate: {args.lr}")
        print(f"Epochs: {args.epochs}")
        print(f"Image size: {args.image_size}x{args.image_size}")
        print(f"Number of classes: {args.num_classes}")
        print("=" * 70 + "\n")

        # Create checkpoint directory
        Path(args.checkpoint_dir).mkdir(parents=True, exist_ok=True)

        # Initialize WandB
        if not args.no_wandb:
            self._init_wandb()

        # Build model
        print("[1/5] Building model...")
        self.model, self.x, self.predictions = build_yolo11n(
            num_classes=args.num_classes, input_size=args.image_size
        )
        print(f"✓ Model built with {len(self.model.params)} parameters")
        total_params = sum(p.get_value().size for p in self.model.params)
        print(f"✓ Total parameters: {total_params:,}\n")

        # Define loss
        print("[2/5] Setting up loss function...")
        self.loss, self.loss_dict = yolo_loss(
            self.predictions, targets=None, num_classes=args.num_classes
        )
        print("✓ Loss function defined\n")

        # Compute gradients
        print("[3/5] Computing gradients...")
        self.grads = self._compute_gradients()
        print(f"✓ Computed gradients for {len(self.grads)} parameters\n")

        # SGD optimizer
        print("[4/5] Setting up optimizer...")
        self.velocities = self._init_optimizer()
        print("✓ SGD optimizer with momentum initialized\n")

        # Compile training function
        print("[5/5] Compiling training function...")
        self.train_fn = self._compile_train_function()
        print("✓ Training function compiled\n")

        # Training state
        self.epoch = 0
        self.best_loss = float("inf")
        self.step = 0

        # Load checkpoint if resuming
        if args.resume:
            self._load_checkpoint(args.resume)

    def _check_gpu(self):
        """Check if GPU is available."""
        try:
            import subprocess

            subprocess.check_output(["nvidia-smi"])
            return True
        except Exception:
            return False

    def _init_wandb(self):
        """Initialize Weights & Biases logging."""
        wandb.init(
            project=self.args.wandb_project,
            entity=self.args.wandb_entity,
            config=vars(self.args),
            name=f"yolo11n_{self.args.image_size}_{time.strftime('%Y%m%d_%H%M%S')}",
        )
        wandb.config.update(
            {
                "total_params": sum(p.get_value().size for p in self.model.params),
                "device": self.device,
            }
        )

    def _compute_gradients(self):
        """Compute gradients for all parameters."""
        grads = []
        for param in self.model.params:
            try:
                grad = pytensor.grad(self.loss, param, disconnected_inputs="ignore")
                grads.append(grad)
            except Exception as e:
                print(f"Warning: Could not compute gradient for {param.name}: {e}")
                grads.append(pt.zeros_like(param))
        return grads

    def _init_optimizer(self):
        """Initialize SGD with momentum optimizer."""
        velocities = []
        for param in self.model.params:
            v = shared(
                np.zeros_like(param.get_value(), dtype="float32"),
                name=f"{param.name}_velocity",
                borrow=True,
            )
            velocities.append(v)
        return velocities

    def _compile_train_function(self):
        """Compile PyTensor function for training step."""
        # Parameter updates with SGD + momentum
        updates = []

        # Hyperparameters as PyTensor scalars with explicit float32 dtype
        momentum = pt.as_tensor_variable(np.float32(self.args.momentum))
        lr = pt.as_tensor_variable(np.float32(self.args.lr))
        weight_decay = pt.as_tensor_variable(np.float32(self.args.weight_decay))

        for param, grad, velocity in zip(
            self.model.params, self.grads, self.velocities
        ):
            # Cast gradient to float32 to match parameters
            grad = pt.cast(grad, "float32")

            # Momentum update: v = momentum * v - lr * grad
            v_new = momentum * velocity - lr * grad

            # Weight decay
            if self.args.weight_decay > 0:
                v_new = v_new - lr * weight_decay * param

            # Parameter update: param = param + v
            p_new = param + v_new

            updates.append((velocity, v_new))
            updates.append((param, p_new))

        # Compile function
        train_fn = function(
            inputs=[self.x],
            outputs=[self.loss, self.loss_dict["box_loss"], self.loss_dict["cls_loss"]],
            updates=updates,
            name="train_step",
        )

        return train_fn

    def train_epoch(self, dataloader):
        """Train for one epoch."""
        epoch_losses = []
        epoch_box_losses = []
        epoch_cls_losses = []

        pbar = tqdm(dataloader, desc=f"Epoch {self.epoch + 1}/{self.args.epochs}")

        for batch_idx, batch in enumerate(pbar):
            # Get batch data
            images = batch["images"]  # (batch, 3, 320, 320)

            # Training step
            loss, box_loss, cls_loss = self.train_fn(images)

            # Record losses
            epoch_losses.append(float(loss))
            epoch_box_losses.append(float(box_loss))
            epoch_cls_losses.append(float(cls_loss))

            # Update progress bar
            pbar.set_postfix(
                {
                    "loss": f"{loss:.4f}",
                    "box": f"{box_loss:.4f}",
                    "cls": f"{cls_loss:.4f}",
                }
            )

            # Log to WandB
            if not self.args.no_wandb and (batch_idx + 1) % self.args.log_every == 0:
                wandb.log(
                    {
                        "train/loss": loss,
                        "train/box_loss": box_loss,
                        "train/cls_loss": cls_loss,
                        "train/learning_rate": self.args.lr,
                        "train/epoch": self.epoch + 1,
                        "train/step": self.step,
                    }
                )

            self.step += 1

        # Compute epoch averages
        avg_loss = np.mean(epoch_losses)
        avg_box_loss = np.mean(epoch_box_losses)
        avg_cls_loss = np.mean(epoch_cls_losses)

        return avg_loss, avg_box_loss, avg_cls_loss

    def save_checkpoint(self, filename=None, is_best=False):
        """Save training checkpoint."""
        if filename is None:
            filename = f"checkpoint_epoch_{self.epoch + 1}.npz"

        checkpoint_path = Path(self.args.checkpoint_dir) / filename

        # Save model parameters
        param_dict = {}
        for i, param in enumerate(self.model.params):
            param_dict[f"param_{i}"] = param.get_value()

        # Save optimizer state
        for i, velocity in enumerate(self.velocities):
            param_dict[f"velocity_{i}"] = velocity.get_value()

        # Save training state
        param_dict["epoch"] = self.epoch + 1
        param_dict["best_loss"] = self.best_loss
        param_dict["step"] = self.step

        np.savez(checkpoint_path, **param_dict)
        print(f"✓ Checkpoint saved: {checkpoint_path}")

        if is_best:
            best_path = Path(self.args.checkpoint_dir) / "best_model.npz"
            np.savez(best_path, **param_dict)
            print(f"✓ Best model saved: {best_path}")

        # Log to WandB
        if not self.args.no_wandb:
            wandb.save(str(checkpoint_path))

    def _load_checkpoint(self, checkpoint_path):
        """Load checkpoint for resuming training."""
        print(f"Loading checkpoint: {checkpoint_path}")
        checkpoint = np.load(checkpoint_path)

        # Load model parameters
        for i, param in enumerate(self.model.params):
            param.set_value(checkpoint[f"param_{i}"])

        # Load optimizer state
        for i, velocity in enumerate(self.velocities):
            velocity.set_value(checkpoint[f"velocity_{i}"])

        # Load training state
        self.epoch = int(checkpoint["epoch"])
        self.best_loss = float(checkpoint["best_loss"])
        self.step = int(checkpoint["step"])

        print(f"✓ Resumed from epoch {self.epoch}")

    def train(self):
        """Main training loop."""
        print("\n" + "=" * 70)
        print("Starting Training".center(70))
        print("=" * 70 + "\n")

        # Load COCO dataset
        try:
            print("Loading COCO training dataset...")
            train_dataset = COCODataset(
                data_dir=self.args.data_dir,
                split="train",
                image_size=self.args.image_size,
                download=False,  # Don't auto-download, let user run setup.sh first
            )

            if len(train_dataset) == 0:
                raise ValueError("No images found in dataset")

            print(f"✓ Loaded {len(train_dataset)} training images\n")
            use_real_data = True

        except Exception as e:
            print(f"Warning: Could not load COCO dataset: {e}")
            print("Falling back to synthetic data for demonstration")
            print(
                "To use real data, run: python dataset.py --data-dir ./data/coco --split train\n"
            )
            use_real_data = False

        for epoch in range(self.epoch, self.args.epochs):
            self.epoch = epoch

            # Create dataloader for this epoch
            if use_real_data:
                dataloader = create_dataloader(
                    train_dataset, batch_size=self.args.batch_size, shuffle=True
                )
            else:
                # Fallback: Generate synthetic batches
                num_batches = 100  # Simulate 100 batches per epoch

                class SyntheticDataloader:
                    def __init__(self, num_batches, batch_size, image_size):
                        self.num_batches = num_batches
                        self.batch_size = batch_size
                        self.image_size = image_size

                    def __iter__(self):
                        for _ in range(self.num_batches):
                            images = (
                                np.random.randn(
                                    self.batch_size, 3, self.image_size, self.image_size
                                ).astype("float32")
                                * 0.1
                            )
                            yield {"images": images}

                    def __len__(self):
                        return self.num_batches

                dataloader = SyntheticDataloader(
                    num_batches, self.args.batch_size, self.args.image_size
                )

            # Train for one epoch
            avg_loss, avg_box_loss, avg_cls_loss = self.train_epoch(dataloader)

            # Print epoch summary
            print(f"\nEpoch {epoch + 1} Summary:")
            print(f"  Average Loss: {avg_loss:.6f}")
            print(f"  Box Loss: {avg_box_loss:.6f}")
            print(f"  Cls Loss: {avg_cls_loss:.6f}")

            # Log epoch metrics to WandB
            if not self.args.no_wandb:
                wandb.log(
                    {
                        "epoch/loss": avg_loss,
                        "epoch/box_loss": avg_box_loss,
                        "epoch/cls_loss": avg_cls_loss,
                        "epoch/number": epoch + 1,
                    }
                )

            # Save checkpoint
            if (epoch + 1) % self.args.save_every == 0:
                self.save_checkpoint()

            # Save best model
            if avg_loss < self.best_loss:
                self.best_loss = avg_loss
                self.save_checkpoint(is_best=True)
                print(f"✓ New best loss: {self.best_loss:.6f}")

            print()

        print("\n" + "=" * 70)
        print("Training Complete!".center(70))
        print("=" * 70)
        print(f"\nBest loss: {self.best_loss:.6f}")
        print(f"Final checkpoint: {self.args.checkpoint_dir}/best_model.npz")

        # Export to ONNX
        if self.args.export_onnx:
            self.export_onnx()

    def export_onnx(self):
        """Export trained model to ONNX format."""
        print("\n" + "=" * 70)
        print("Exporting to ONNX".center(70))
        print("=" * 70 + "\n")

        try:
            from pytensor.link.onnx import export_onnx

            onnx_path = Path(self.args.checkpoint_dir) / "yolo11n_best.onnx"

            # Export model
            export_onnx(
                inputs=[self.x],
                outputs=[
                    self.predictions["p3"],
                    self.predictions["p4"],
                    self.predictions["p5"],
                ],
                filename=str(onnx_path),
            )

            print(f"✓ ONNX model exported: {onnx_path}")
            print("\nTo download from server:")
            print(f"  scp user@server:{onnx_path.absolute()} .")

            if not self.args.no_wandb:
                wandb.save(str(onnx_path))

        except Exception as e:
            print(f"Error exporting to ONNX: {e}")
            print("You can export manually after training using:")
            print("  from pytensor.link.onnx import export_onnx")


def main():
    """Main training entry point."""
    args = parse_args()

    # Create trainer
    trainer = Trainer(args)

    # Train
    try:
        trainer.train()
    except KeyboardInterrupt:
        print("\n\nTraining interrupted by user")
        print("Saving checkpoint...")
        trainer.save_checkpoint(filename="interrupted_checkpoint.npz")
        print("✓ Checkpoint saved")
    except Exception as e:
        print(f"\n\nError during training: {e}")
        import traceback

        traceback.print_exc()
        print("\nSaving emergency checkpoint...")
        trainer.save_checkpoint(filename="error_checkpoint.npz")
        raise
    finally:
        if not args.no_wandb:
            wandb.finish()


if __name__ == "__main__":
    main()
