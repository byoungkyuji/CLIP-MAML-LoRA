import os
import pickle
import random
from collections import OrderedDict, defaultdict

from dassl.data.datasets import DATASET_REGISTRY, Datum, DatasetBase
from dassl.utils import listdir_nohidden, mkdir_if_missing

@DATASET_REGISTRY.register()
class MetaImageNet(DatasetBase):
    template = ['a photo of a {}.']
    dataset_dir = "imagenet"

    def __init__(self, cfg):
        root = os.path.abspath(os.path.expanduser(cfg.DATASET.ROOT))
        self.dataset_dir = os.path.join(root, self.dataset_dir)
        self.image_dir = os.path.join(self.dataset_dir, "images")
        self.preprocessed = os.path.join(self.dataset_dir, "preprocessed.pkl")
        self.split_fewshot_dir = os.path.join(self.dataset_dir, "split_fewshot")
        mkdir_if_missing(self.split_fewshot_dir)

        train, test = self.load_or_preprocess_data()

        # Meta-train / Meta-test 분할
        meta_train, meta_test = self.split_meta_train_test(
            train, test, cfg.DATASET.META_TRAIN_SPLIT
        )

        # Support / Query 분할
        self.meta_train_tasks = self.create_meta_tasks(
            meta_train, cfg.DATASET.NUM_SHOTS, cfg.DATASET.QUERY_SIZE
        )
        self.meta_test_tasks = self.create_meta_tasks(
            meta_test, cfg.DATASET.NUM_SHOTS, cfg.DATASET.QUERY_SIZE
        )

    def load_or_preprocess_data(self):
        """Load or preprocess data for train and test."""
        if os.path.exists(self.preprocessed):
            with open(self.preprocessed, "rb") as f:
                preprocessed = pickle.load(f)
                return preprocessed["train"], preprocessed["test"]

        text_file = os.path.join(self.dataset_dir, "classnames.txt")
        classnames = self.read_classnames(text_file)
        train = self.read_data(classnames, "train")
        test = self.read_data(classnames, "val")

        preprocessed = {"train": train, "test": test}
        with open(self.preprocessed, "wb") as f:
            pickle.dump(preprocessed, f, protocol=pickle.HIGHEST_PROTOCOL)

        return train, test

    def split_meta_train_test(self, train, test, split_ratio=0.8):
        """Split train and test data into meta-train and meta-test classes."""
        class_labels = list(set(item.label for item in train))
        random.shuffle(class_labels)

        num_train_classes = int(len(class_labels) * split_ratio)
        meta_train_classes = set(class_labels[:num_train_classes])
        meta_test_classes = set(class_labels[num_train_classes:])

        return (
            [item for item in train if item.label in meta_train_classes],
            [item for item in test if item.label in meta_test_classes],
        )

    def create_meta_tasks(self, data, num_shots, query_size):
        """Create meta-learning tasks by splitting classes into support and query sets."""
        grouped_data = self.group_by_class(data)
        tasks = []

        for label, items in grouped_data.items():
            if len(items) < num_shots + query_size:
                continue  # Skip classes with insufficient data
            random.shuffle(items)

            support_set = items[:num_shots]
            query_set = items[num_shots:num_shots + query_size]

            tasks.append({"support": support_set, "query": query_set})

        return tasks

    def group_by_class(self, data):
        """Group data by class labels."""
        grouped = defaultdict(list)
        for item in data:
            grouped[item.label].append(item)
        return grouped

    @staticmethod
    def read_classnames(text_file):
        """Read class names from a text file."""
        classnames = OrderedDict()
        with open(text_file, "r") as f:
            for line in f:
                line = line.strip().split(" ")
                folder, classname = line[0], " ".join(line[1:])
                classnames[folder] = classname
        return classnames

    def read_data(self, classnames, split_dir):
        """Read data from the specified split directory."""
        split_dir = os.path.join(self.image_dir, split_dir)
        folders = sorted(f.name for f in os.scandir(split_dir) if f.is_dir())
        items = []

        for label, folder in enumerate(folders):
            imnames = listdir_nohidden(os.path.join(split_dir, folder))
            classname = classnames[folder]
            for imname in imnames:
                impath = os.path.join(split_dir, folder, imname)
                items.append(Datum(impath=impath, label=label, classname=classname))

        return items
