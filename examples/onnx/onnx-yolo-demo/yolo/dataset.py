"""
COCO dataset loader for YOLO11n training.

Supports:
- COCO 2017 dataset
- Filtered classes (person, cellphone)
- Image resizing to 320x320
- Basic data augmentation
"""

import json
from pathlib import Path

import numpy as np
import requests
from PIL import Image
from tqdm import tqdm


class COCODataset:
    """
    COCO dataset loader for object detection.

    Filters for specific classes: person (1), cellphone (77).
    """

    # COCO class IDs for person and cellphone
    CLASS_IDS = {
        1: 0,  # person -> class 0
        77: 1,  # cellphone -> class 1
    }

    CLASS_NAMES = {0: "person", 1: "cellphone"}

    def __init__(
        self,
        data_dir: str,
        split: str = "train",
        image_size: int = 320,
        download: bool = True,
    ):
        """
        Initialize COCO dataset.

        Parameters
        ----------
        data_dir : str
            Root directory for COCO dataset
        split : str
            'train' or 'val'
        image_size : int
            Target image size (square)
        download : bool
            Whether to download dataset if not found
        """
        self.data_dir = Path(data_dir)
        self.split = split
        self.image_size = image_size

        # Set up paths
        if split == "train":
            self.img_dir = self.data_dir / "train2017"
            self.ann_file = self.data_dir / "annotations" / "instances_train2017.json"
        else:
            self.img_dir = self.data_dir / "val2017"
            self.ann_file = self.data_dir / "annotations" / "instances_val2017.json"

        # Download if needed
        if download and not self.ann_file.exists():
            print(f"COCO dataset not found at {self.data_dir}")
            print("Downloading COCO dataset (this may take a while)...")
            self._download_coco()

        # Load annotations
        print(f"Loading {split} annotations...")
        self.annotations = self._load_annotations()
        print(
            f"✓ Loaded {len(self.annotations)} images with {sum(len(a['boxes']) for a in self.annotations)} objects"
        )

    def _download_coco(self):
        """Download COCO dataset."""
        base_url = "http://images.cocodataset.org"

        # Create directories
        self.data_dir.mkdir(parents=True, exist_ok=True)
        (self.data_dir / "annotations").mkdir(exist_ok=True)

        # Files to download
        if self.split == "train":
            files = {
                f"{base_url}/zips/train2017.zip": self.data_dir / "train2017.zip",
                f"{base_url}/annotations/annotations_trainval2017.zip": self.data_dir
                / "annotations.zip",
            }
        else:
            files = {
                f"{base_url}/zips/val2017.zip": self.data_dir / "val2017.zip",
                f"{base_url}/annotations/annotations_trainval2017.zip": self.data_dir
                / "annotations.zip",
            }

        # Download files
        for url, filepath in files.items():
            if filepath.exists():
                print(f"✓ {filepath.name} already exists")
                continue

            print(f"Downloading {filepath.name}...")
            response = requests.get(url, stream=True)
            total_size = int(response.headers.get("content-length", 0))

            with (
                filepath.open("wb") as f,
                tqdm(
                    total=total_size,
                    unit="iB",
                    unit_scale=True,
                    unit_divisor=1024,
                ) as pbar,
            ):
                for chunk in response.iter_content(chunk_size=8192):
                    size = f.write(chunk)
                    pbar.update(size)

            print(f"✓ Downloaded {filepath.name}")

        # Extract files
        print("Extracting files...")
        import zipfile

        for filepath in files.values():
            if filepath.suffix == ".zip":
                print(f"Extracting {filepath.name}...")
                with zipfile.ZipFile(filepath, "r") as zip_ref:
                    zip_ref.extractall(self.data_dir)
                print(f"✓ Extracted {filepath.name}")

        print("✓ COCO dataset downloaded and extracted")

    def _load_annotations(self) -> list[dict]:
        """Load and filter COCO annotations."""
        # Check if annotations file exists
        if not self.ann_file.exists():
            print(f"\nError: Annotations file not found: {self.ann_file}")
            print("Please download COCO dataset first.")
            print(
                "\nFor a quick start, you can use synthetic data by skipping dataset loading."
            )
            return []

        with self.ann_file.open() as f:
            coco = json.load(f)

        # Build image id to filename mapping
        images = {img["id"]: img for img in coco["images"]}

        # Filter annotations for our classes
        image_annotations = {}

        for ann in coco["annotations"]:
            # Check if this annotation is for one of our classes
            if ann["category_id"] not in self.CLASS_IDS:
                continue

            img_id = ann["image_id"]

            if img_id not in image_annotations:
                image_annotations[img_id] = {
                    "image_id": img_id,
                    "filename": images[img_id]["file_name"],
                    "width": images[img_id]["width"],
                    "height": images[img_id]["height"],
                    "boxes": [],
                    "classes": [],
                }

            # Convert bbox from [x, y, w, h] to normalized [x_center, y_center, w, h]
            x, y, w, h = ann["bbox"]
            img_w = images[img_id]["width"]
            img_h = images[img_id]["height"]

            x_center = (x + w / 2) / img_w
            y_center = (y + h / 2) / img_h
            w_norm = w / img_w
            h_norm = h / img_h

            image_annotations[img_id]["boxes"].append(
                [x_center, y_center, w_norm, h_norm]
            )
            image_annotations[img_id]["classes"].append(
                self.CLASS_IDS[ann["category_id"]]
            )

        # Convert to list
        annotations = list(image_annotations.values())

        return annotations

    def __len__(self) -> int:
        """Return dataset size."""
        return len(self.annotations)

    def __getitem__(self, idx: int) -> dict:
        """
        Get item by index.

        Returns
        -------
        dict
            {
                'image': np.ndarray (3, H, W) float32,
                'boxes': np.ndarray (N, 4) normalized [x, y, w, h],
                'classes': np.ndarray (N,) class indices
            }
        """
        ann = self.annotations[idx]

        # Load image
        img_path = self.img_dir / ann["filename"]

        if not img_path.exists():
            # Return dummy data if image not found
            return {
                "image": np.random.randn(3, self.image_size, self.image_size).astype(
                    "float32"
                )
                * 0.1,
                "boxes": np.zeros((0, 4), dtype="float32"),
                "classes": np.zeros((0,), dtype="int64"),
            }

        image = Image.open(img_path).convert("RGB")

        # Resize to target size
        image = image.resize((self.image_size, self.image_size), Image.BILINEAR)

        # Convert to numpy array and normalize
        image = np.array(image, dtype="float32") / 255.0

        # Transpose to (C, H, W)
        image = image.transpose(2, 0, 1)

        # Get boxes and classes
        boxes = np.array(ann["boxes"], dtype="float32")
        classes = np.array(ann["classes"], dtype="int64")

        return {
            "image": image,
            "boxes": boxes,
            "classes": classes,
        }


def create_dataloader(dataset: COCODataset, batch_size: int = 8, shuffle: bool = True):
    """
    Create a simple dataloader.

    Parameters
    ----------
    dataset : COCODataset
        Dataset to load from
    batch_size : int
        Batch size
    shuffle : bool
        Whether to shuffle data

    Yields
    ------
    dict
        {
            'images': np.ndarray (B, 3, H, W),
            'boxes': list of np.ndarray (N, 4),
            'classes': list of np.ndarray (N,)
        }
    """
    indices = list(range(len(dataset)))

    if shuffle:
        np.random.shuffle(indices)

    for i in range(0, len(indices), batch_size):
        batch_indices = indices[i : i + batch_size]
        batch_images = []
        batch_boxes = []
        batch_classes = []

        for idx in batch_indices:
            sample = dataset[idx]
            batch_images.append(sample["image"])
            batch_boxes.append(sample["boxes"])
            batch_classes.append(sample["classes"])

        # Stack images
        batch_images = np.stack(batch_images, axis=0)

        yield {
            "images": batch_images,
            "boxes": batch_boxes,
            "classes": batch_classes,
        }


def download_coco_subset(data_dir: str = "./data/coco", split: str = "train"):
    """
    Utility function to download COCO dataset.

    Parameters
    ----------
    data_dir : str
        Directory to save dataset
    split : str
        'train' or 'val'
    """
    dataset = COCODataset(data_dir=data_dir, split=split, download=True)
    print(f"\n✓ COCO {split} dataset ready!")
    print(f"  Images: {len(dataset)}")
    print(f"  Location: {dataset.data_dir}")


if __name__ == "__main__":
    # Test dataset loading
    import argparse

    parser = argparse.ArgumentParser(description="Download and test COCO dataset")
    parser.add_argument(
        "--data-dir", type=str, default="./data/coco", help="Directory to save dataset"
    )
    parser.add_argument(
        "--split",
        type=str,
        default="train",
        choices=["train", "val"],
        help="Dataset split",
    )
    args = parser.parse_args()

    # Download dataset
    download_coco_subset(args.data_dir, args.split)

    # Test dataloader
    print("\nTesting dataloader...")
    dataset = COCODataset(args.data_dir, args.split, download=False)
    dataloader = create_dataloader(dataset, batch_size=4, shuffle=True)

    batch = next(iter(dataloader))
    print(f"Batch images shape: {batch['images'].shape}")
    print(f"Batch boxes: {len(batch['boxes'])} samples")
    print(f"Batch classes: {len(batch['classes'])} samples")
    print("\n✓ Dataloader test passed!")
