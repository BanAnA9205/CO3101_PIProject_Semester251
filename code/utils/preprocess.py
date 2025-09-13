import kagglehub
import os
import random
import shutil
import pandas as pd
from tqdm import tqdm

def preprocess( DATASET_NAME: str = "abdallahalidev/plantvillage-dataset",
                plants_to_keep: str | list = "all",
                target_path: str = None,
                train_size: float = 0.7,
                val_size: float = 0.15,
                test_size: float = 0.15,
                seed: int = 42
              ) -> None:
    """
    Downloads and curates the PlantVillage dataset from Kaggle.
    Splits the dataset into training, validation, and test sets based on specified sizes.
    Args:
        DATASET_NAME (str): The Kaggle dataset identifier.
        plants_to_keep (str | list): List of plant names to keep or "all" to keep all plants.
        target_path (str): Path to download and curate the dataset files. If None, prompts user input.
        train_size (float): Proportion of data to use for training set.
        val_size (float): Proportion of data to use for validation set.
        test_size (float): Proportion of data to use for test set.
        seed (int): Random seed for reproducibility.
    """

    assert abs(train_size + val_size + test_size - 1.0) < 1e-6, "Train, val, and test sizes must sum to 1."

    if target_path is None:
        print("Please specify the path to download the dataset files (or leave empty to use current directory):")
        target_path = input().strip()

    if not target_path:
        print("No path provided, using current working directory.")
        target_path = os.getcwd()
    else:
        if not os.path.exists(target_path):
            print(f"The specified path '{target_path}' does not exist. Creating it...")
            os.makedirs(target_path)

    try:
        ds_path = kagglehub.dataset_download(DATASET_NAME)
        print("Dataset downloaded to KaggleHub cache:", ds_path)

        # Only copy the color images directory (i.e. the 'color' folder)
        print(f"Copying dataset files to: {target_path}")
        dataset = os.listdir(ds_path)[0]
        src = os.path.join(ds_path, dataset, "color")
        dest = os.path.join(target_path, dataset)
        train_dir = os.path.join(dest, "train")
        val_dir = os.path.join(dest, "val")
        test_dir = os.path.join(dest, "test")
        df_dir = os.path.join(dest, "dataframes")

        for dir_path in [train_dir, val_dir, test_dir, df_dir]:
            os.makedirs(dir_path, exist_ok=True)

        df_train = pd.DataFrame(columns=["file_name", "plant", "disease"])
        df_val = pd.DataFrame(columns=["file_name", "plant", "disease"])
        df_test = pd.DataFrame(columns=["file_name", "plant", "disease"])

        plants_to_keep = [p.lower() for p in plants_to_keep] if isinstance(plants_to_keep, list) else plants_to_keep.lower()

        for item in tqdm(os.listdir(src), desc="Processing classes"):
            plant, disease = item.split("___")
            if plants_to_keep != "all" and plant.lower() not in plants_to_keep:
                continue

            images = os.listdir(os.path.join(src, item))
            random.seed(seed)
            random.shuffle(images)

            n_total = len(images)
            n_train = int(n_total * train_size)
            n_val = int(n_total * val_size)

            train_images = images[:n_train]
            val_images = images[n_train:n_train + n_val]
            test_images = images[n_train + n_val:]

            for img in train_images:
                new_name, _ = img.split("___")
                new_name += ".jpg"
                shutil.copy(os.path.join(src, item, img), os.path.join(train_dir, new_name))
                df_train = pd.concat([df_train, pd.DataFrame({"file_name": [new_name], "plant": [plant], "disease": [disease]})], ignore_index=True)

            for img in val_images:
                new_name, _ = img.split("___")
                new_name += ".jpg"
                shutil.copy(os.path.join(src, item, img), os.path.join(val_dir, new_name))
                df_val = pd.concat([df_val, pd.DataFrame({"file_name": [new_name], "plant": [plant], "disease": [disease]})], ignore_index=True)

            for img in test_images:
                new_name, _ = img.split("___")
                new_name += ".jpg"
                shutil.copy(os.path.join(src, item, img), os.path.join(test_dir, new_name))
                df_test = pd.concat([df_test, pd.DataFrame({"file_name": [new_name], "plant": [plant], "disease": [disease]})], ignore_index=True)

        # Reshuffle dataframes
        df_train = df_train.sample(frac=1, random_state=seed).reset_index(drop=True)
        df_val = df_val.sample(frac=1, random_state=seed).reset_index(drop=True)
        df_test = df_test.sample(frac=1, random_state=seed).reset_index(drop=True)

        df_train.to_csv(os.path.join(df_dir, "train_labels.csv"), index=False)
        df_val.to_csv(os.path.join(df_dir, "val_labels.csv"), index=False)
        df_test.to_csv(os.path.join(df_dir, "test_labels.csv"), index=False)

        print(f"Dataset files copied to: {dest}")
        print("Curating dataset to keep only specified plants:", plants_to_keep)

    except Exception as e:
        print("An error occurred while downloading or curating the dataset:", e)

if __name__ == "__main__":
    preprocess(plants_to_keep=["apple", "corn", "grape", "peach", "potato", "strawberry"])