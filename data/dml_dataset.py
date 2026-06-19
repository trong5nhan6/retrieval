"""
Deep-metric-learning datasets: CUB-200-2011, Cars196, In-Shop.

Standard zero-shot retrieval protocol (train on first half of classes, test on
the unseen second half). Each dataset is trained SEPARATELY (its own model),
matching published SOTA tables.

Expected directory layout (set roots in config.HCFG.data_roots). The parsers
auto-detect the common on-disk variants, so the roots below are what we actually
ship in `datasets/`:

  CUB_200_2011/                    (a nested CUB_200_2011/ level is auto-detected)
    images.txt
    image_class_labels.txt
    images/<class>/<file>.jpg

  Cars196/                         (Kaggle CSV layout)
    anno_train.csv                 'file,x1,y1,x2,y2,class' (class 1..196)
    anno_test.csv                  same format
    names.csv                      196 class names (line i == class i)
    car_data/car_data/train/<ClassName>/<file>.jpg
    car_data/car_data/test/<ClassName>/<file>.jpg
  cars196/                         (Stanford .mat layout — still supported)
    cars_annos.mat
    car_ims/<file>.jpg

  In-shop Clothes Retrieval Benchmark/   (DeepFashion In-Shop)
    Eval/list_eval_partition.txt   (root-level file is also accepted)
    Img/img/<...>.jpg              (extract Img/img.zip first!)

Usage:
  from data.dml_dataset import get_dml_loaders
  loaders = get_dml_loaders("cub", HCFG)        # dict
  # CUB/Cars: {"train":..., "test":...}
  # In-Shop : {"train":..., "query":..., "gallery":...}
"""
import os
import platform
from collections import defaultdict

import torch
from torch.utils.data import Dataset, DataLoader, Sampler
from torchvision import transforms
from PIL import Image

_NUM_WORKERS = 0 if platform.system() == "Windows" else 4
_MEAN = (0.485, 0.456, 0.406)
_STD = (0.229, 0.224, 0.225)


# ── transforms (standard DML: RRC train, Resize+CenterCrop test) ────────────
def _tf(image_size, train):
    if train:
        return transforms.Compose([
            transforms.RandomResizedCrop(image_size, scale=(0.5, 1.0)),
            transforms.RandomHorizontalFlip(),
            transforms.ToTensor(), transforms.Normalize(_MEAN, _STD),
        ])
    return transforms.Compose([
        transforms.Resize(int(image_size * 1.14)),
        transforms.CenterCrop(image_size),
        transforms.ToTensor(), transforms.Normalize(_MEAN, _STD),
    ])


# ── generic list-based dataset ──────────────────────────────────────────────
class _ListDataset(Dataset):
    """items: list of (abs_path, int_label)."""
    def __init__(self, items, image_size, train):
        self.items = items
        self.tf = _tf(image_size, train)

    def __len__(self):
        return len(self.items)

    def __getitem__(self, idx):
        path, label = self.items[idx]
        img = Image.open(path).convert("RGB")
        return self.tf(img), label


# ── class-balanced sampler (P classes x K samples) ──────────────────────────
class ClassBalancedSampler(Sampler):
    def __init__(self, labels, classes_per_batch, samples_per_class, seed=42):
        self.labels = labels
        self.P = classes_per_batch
        self.K = samples_per_class
        self.seed = seed
        self.by_class = defaultdict(list)
        for i, y in enumerate(labels):
            self.by_class[int(y)].append(i)
        self.classes = list(self.by_class.keys())
        self.num_batches = max(1, len(labels) // (self.P * self.K))

    def __len__(self):
        return self.num_batches * self.P * self.K

    def __iter__(self):
        g = torch.Generator().manual_seed(self.seed)
        for _ in range(self.num_batches):
            cls = torch.randperm(len(self.classes), generator=g)[:self.P]
            for c in cls:
                pool = self.by_class[self.classes[int(c)]]
                if len(pool) >= self.K:
                    pick = torch.randperm(len(pool), generator=g)[:self.K]
                else:
                    pick = torch.randint(len(pool), (self.K,), generator=g)
                for p in pick:
                    yield pool[int(p)]


# ── parsers ─────────────────────────────────────────────────────────────────
def _parse_cub(root):
    # The Kaggle/CUB tarball nests the real data one extra level deep.
    if not os.path.exists(os.path.join(root, "images.txt")):
        nested = os.path.join(root, "CUB_200_2011")
        if os.path.exists(os.path.join(nested, "images.txt")):
            root = nested
    with open(os.path.join(root, "images.txt")) as f:
        id2path = {int(l.split()[0]): l.split()[1] for l in f}
    with open(os.path.join(root, "image_class_labels.txt")) as f:
        id2cls = {int(l.split()[0]): int(l.split()[1]) for l in f}
    train, test = [], []
    for iid, rel in id2path.items():
        cls = id2cls[iid]                          # 1..200
        ap = os.path.join(root, "images", rel)
        # Zero-shot split: first 100 classes train, last 100 test.
        (train if cls <= 100 else test).append((ap, cls - 1))
    return {"train": train, "test": test}, 100


def _parse_cars(root):
    """Cars196 — Kaggle CSV layout (preferred) with .mat fallback.

    The Kaggle release ships the original classification split (train/ and test/
    each contain all 196 classes). We pool everything and re-split by *class* to
    follow the standard zero-shot retrieval protocol (classes 1..98 train,
    99..196 test).
    """
    csv_train = os.path.join(root, "anno_train.csv")
    if os.path.exists(csv_train):
        import csv as _csv
        # names.csv: line i (1-based) holds the folder name for class i.
        # Slashes in a class name (e.g. 'Ram C/V ...') become '-' on disk.
        with open(os.path.join(root, "names.csv")) as f:
            class_names = [ln.strip().replace("/", "-") for ln in f if ln.strip()]

        items_all = []
        for split, csv_name in (("train", "anno_train.csv"),
                                ("test", "anno_test.csv")):
            split_dir = os.path.join(root, "car_data", "car_data", split)
            with open(os.path.join(root, csv_name)) as f:
                for row in _csv.reader(f):
                    if not row:
                        continue
                    fname, cls = row[0], int(row[-1])      # 'file,x1,y1,x2,y2,class'
                    ap = os.path.join(split_dir, class_names[cls - 1], fname)
                    items_all.append((ap, cls))
    else:
        import scipy.io as sio
        anno = sio.loadmat(os.path.join(root, "cars_annos.mat"))["annotations"][0]
        items_all = []
        for a in anno:
            rel = str(a[0][0])                              # e.g. 'car_ims/000001.jpg'
            cls = int(a[5][0][0])                           # 1..196
            items_all.append((os.path.join(root, rel), cls))

    train = [(p, c - 1) for p, c in items_all if c <= 98]
    test = [(p, c - 1) for p, c in items_all if c > 98]
    return {"train": train, "test": test}, 98


def _parse_inshop(root):
    # Partition file may sit at the root or under Eval/.
    part = os.path.join(root, "list_eval_partition.txt")
    if not os.path.exists(part):
        part = os.path.join(root, "Eval", "list_eval_partition.txt")
    # Image paths in the partition file are 'img/...'; locate the dir holding 'img/'.
    img_root = root
    for cand in (root, os.path.join(root, "Img")):
        if os.path.isdir(os.path.join(cand, "img")):
            img_root = cand
            break

    train, query, gallery = [], [], []
    label_map = {}
    with open(part) as f:
        lines = f.read().splitlines()
    # first line = count, second = header
    for l in lines[2:]:
        parts = l.split()
        if len(parts) < 3:
            continue
        rel, item_id, status = parts[0], parts[1], parts[2]
        if item_id not in label_map:
            label_map[item_id] = len(label_map)
        lbl = label_map[item_id]
        ap = os.path.join(img_root, rel)
        if status == "train":
            train.append((ap, lbl))
        elif status == "query":
            query.append((ap, lbl))
        elif status == "gallery":
            gallery.append((ap, lbl))
    return {"train": train, "query": query, "gallery": gallery}, len(label_map)


_PARSERS = {"cub": _parse_cub, "cars": _parse_cars, "inshop": _parse_inshop}


# ── public API ──────────────────────────────────────────────────────────────
def get_dml_loaders(name: str, cfg):
    name = name.lower()
    root = cfg.data_roots[name]
    splits, _ = _PARSERS[name](root)
    loaders = {}

    # train loader (class-balanced)
    tr_items = splits["train"]
    tr_ds = _ListDataset(tr_items, cfg.image_size, train=True)
    tr_labels = [l for _, l in tr_items]
    sampler = ClassBalancedSampler(tr_labels, cfg.classes_per_batch,
                                   cfg.samples_per_class, seed=cfg.seed)
    loaders["train"] = DataLoader(
        tr_ds, batch_size=cfg.classes_per_batch * cfg.samples_per_class,
        sampler=sampler, num_workers=_NUM_WORKERS, pin_memory=True, drop_last=True)

    # eval loaders
    eval_splits = ["test"] if name in ("cub", "cars") else ["query", "gallery"]
    for sp in eval_splits:
        ds = _ListDataset(splits[sp], cfg.image_size, train=False)
        loaders[sp] = DataLoader(ds, batch_size=128, shuffle=False,
                                 num_workers=_NUM_WORKERS, pin_memory=True)
    return loaders
