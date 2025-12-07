import os
from torch.utils.data import Dataset
from PIL import Image
import torchvision.transforms as T
import pandas as pd

class ImgDataset(Dataset):
    def __init__(
        self,
        df: pd.DataFrame,
        img_dir: str,
        mode: str = "RGB",
        transform: T.Compose | None = None,
        classes: list | None = None
    ):
        """
        Custom image dataset that maps (plant, disease) pairs to integer class labels.

        Args:
            df (pd.DataFrame):
                DataFrame with columns ['file_name', 'plant', 'disease'].
            img_dir (str):
                Path to the directory containing image files.
            mode (str, optional):
                Image color mode, e.g., "RGB" or "L". Defaults to "RGB".
            transform (torchvision.transforms.Compose, optional):
                Transformations applied to each image. Defaults to `ToTensor()`.
            classes (list, optional):
                List of unique (plant, disease) pairs. If None, it will be
                generated from the DataFrame. Defaults to None.
        """
        # csv should have 3 columns: file_name, plant, disease
        assert df is not None, "Dataframe is None"
        assert "file_name" in df.columns, "'file_name' column not in dataframe"
        assert "plant" in df.columns, "'plant' column not in dataframe"
        assert "disease" in df.columns, "'disease' column not in dataframe"
        
        self.df = df
        self.mode = mode
        self.img_dir = img_dir
        self.transform = transform if transform is not None else T.ToTensor()

        # each class is a unique (plant, disease) pair
        if classes is not None:
            self.classes = classes
        else:
            self.classes = (
                self.df[["plant", "disease"]]
                .drop_duplicates()
                .sort_values(["plant", "disease"], ignore_index=True)
            )

        # a mapping from (plant, disease) to class index
        self.class_to_idx = {
            (self.classes.iloc[i, 0], self.classes.iloc[i, 1]): i 
            for i in range(len(self.classes))
        }

    def __len__(self):
        return len(self.df)

    def __getitem__(self, idx):
        img_path = os.path.join(self.img_dir, self.df.iloc[idx, 0])
        image = Image.open(img_path).convert(self.mode)
        label = self.class_to_idx[(self.df.iloc[idx, 1], self.df.iloc[idx, 2])]

        # torch expects a tensor
        image = self.transform(image)

        return image, label