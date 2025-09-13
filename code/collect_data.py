import kagglehub
import os
import shutil

def collect_data(DATASET_NAME = "abdallahalidev/plantvillage-dataset",
                 plants_to_keep = ["apple", "corn", "grape", "peach", "potato", "strawberry"]):
    
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
        src = os.path.join(ds_path, dataset)
        dest = os.path.join(target_path, dataset)
        for item in os.listdir(src):
            s = os.path.join(src, item)
            d = os.path.join(dest, item)
            if os.path.isdir(s) and item == "color":
                # Copy only the plants we want to keep
                for plant_name in os.listdir(s):
                    if any(plant in plant_name.lower() for plant in plants_to_keep):
                        shutil.copytree(os.path.join(s, plant_name), os.path.join(d, plant_name))

        print(f"Dataset files copied to: {target_path}")
        print("Curating dataset to keep only specified plants:", plants_to_keep)

    except Exception as e:
        print("An error occurred while downloading or curating the dataset:", e)

if __name__ == "__main__":
    collect_data()