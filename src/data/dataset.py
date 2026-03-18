import os
from PIL import Image
from torch.utils.data import Dataset

from .roi import crop_canine_roi
from .transforms import get_train_transforms, get_eval_transforms


class OPGDataset(Dataset):
    EXTS = {".jpg", ".jpeg", ".png", ".bmp", ".tiff", ".tif"}

    def __init__(
        self,
        root: str,
        mode: str = "train",
        img_size: int = 224,
        augment_factor: int = 20,
        use_roi_crop: bool = True,
    ):
        self.root = root
        self.mode = mode
        self.img_size = img_size
        self.augment_factor = augment_factor if mode == "train" else 1
        self.use_roi_crop = use_roi_crop

        self.transform = (
            get_train_transforms(img_size) if mode == "train"
            else get_eval_transforms(img_size)
        )

        self.samples = []
        self._load_samples()

    def _load_samples(self):
        pos_dir = os.path.join(self.root, "positive")
        neg_dir = os.path.join(self.root, "negative")

        if os.path.isdir(pos_dir):
            for f in os.listdir(pos_dir):
                if os.path.splitext(f)[1].lower() in self.EXTS:
                    self.samples.append((os.path.join(pos_dir, f), 1))

        if self.mode == "eval" and os.path.isdir(neg_dir):
            for f in os.listdir(neg_dir):
                if os.path.splitext(f)[1].lower() in self.EXTS:
                    self.samples.append((os.path.join(neg_dir, f), 0))

        if not self.samples:
            raise FileNotFoundError(
                f"No images found in {self.root}. "
                "Expected subfolders: positive/ [and negative/ for eval]"
            )

        print(
            f"[Dataset] mode={self.mode} | "
            f"positives={sum(1 for _, l in self.samples if l == 1)} | "
            f"negatives={sum(1 for _, l in self.samples if l == 0)} | "
            f"effective size={len(self)} (x{self.augment_factor} aug)"
        )

    def __len__(self):
        return len(self.samples) * self.augment_factor

    def __getitem__(self, idx):
        path, label = self.samples[idx % len(self.samples)]
        img = Image.open(path).convert("L")

        if self.use_roi_crop:
            img = crop_canine_roi(img)

        img = self.transform(img)
        return img, label, path
