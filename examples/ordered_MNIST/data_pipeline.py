import shutil
from pathlib import Path

import ml_confs
from datasets import DatasetDict, interleave_datasets, load_dataset

main_path = Path(__file__).parent
data_path = main_path / "__data__"
configs = ml_confs.from_file(main_path / "configs.yaml")


def make_dataset():
    # Data pipeline
    MNIST = load_dataset("mnist", keep_in_memory=True)
    digit_ds = []
    for i in range(configs.classes):
        digit_ds.append(
            MNIST.filter(
                lambda example: example["label"] == i, keep_in_memory=True, num_proc=8
            )
        )
    ordered_MNIST = DatasetDict()
    # Order the digits in the dataset and select only a subset of the data
    for split in ["train", "test"]:
        ordered_MNIST[split] = interleave_datasets(
            [ds[split] for ds in digit_ds], split=split
        ).select(range(configs[f"{split}_samples"]))
    _tmp_ds = ordered_MNIST["train"].train_test_split(
        test_size=configs.val_ratio, shuffle=False
    )
    ordered_MNIST["train"] = _tmp_ds["train"]
    ordered_MNIST["validation"] = _tmp_ds["test"]
    ordered_MNIST.set_format(type="torch", columns=["image", "label"])
    ordered_MNIST = ordered_MNIST.map(
        lambda example: {"image": example["image"] / 255.0, "label": example["label"]},
        batched=True,
        keep_in_memory=True,
        num_proc=2,
    )
    ordered_MNIST.save_to_disk(data_path)
    configs.to_file(data_path / "configs.yaml")


def main():
    # Check if data_path exists, if not preprocess the data
    if not data_path.exists():
        print("Data directory not found, preprocessing data.")
        make_dataset()
    else:
        # Try to load the configs.yaml file and compare with the current one, if different, wipe the data_path and preprocess the data
        _saved_configs = ml_confs.from_file(data_path / "configs.yaml")
        configs_changed = False
        for attr in ["train_samples", "test_samples", "classes", "val_ratio"]:
            if _saved_configs[attr] != configs[attr]:
                configs_changed = True
        if configs_changed:
            print("Configs changed, preprocessing data.")
            # Delete the data_path and preprocess the data
            shutil.rmtree(data_path)
            make_dataset()
        else:
            print("Data already preprocessed.")


if __name__ == "__main__":
    main()
