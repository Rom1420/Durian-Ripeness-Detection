import shutil
import os

# Only valid ripeness classes are considered
VALID_CLASSES = {"Ripe1", "Ripe2", "Ripe3", "Ripe4"}


def extract_class(filename):
    """
    Extracts the ripeness class from the filename.
    Expects format: something_ClassX_RipeY_...jpg
    """
    try:
        return filename.split("_")[2]  # e.g., 'Ripe3'
    except IndexError:
        print(f"[!] Incorrect filename format: {filename}")
        return None


def create_dataset_structure():
    """
    Organizes the dataset into three separate folders:
    - 'original': original full-size images
    - 'crop': YOLO-cropped images
    - 'mixed': a combination of original and cropped images

    Each set is organized by ripeness class.
    """
    base = "database"
    crop_dir = os.path.join(base, "crop")
    output_sets = {
        "original": os.path.join("dbsets", "original"),
        "crop": os.path.join("dbsets", "crop"),
        "mixed": os.path.join("dbsets", "mixed"),
    }

    # Create the target folders for each dataset and ripeness class
    for set_path in output_sets.values():
        for ripe in VALID_CLASSES:
            dir_path = os.path.join(set_path, ripe)
            os.makedirs(dir_path, exist_ok=True)

    # Copy original (non-cropped) images to 'original' and 'mixed' folders
    for fname in os.listdir(base):
        if fname.lower().endswith((".jpg", ".jpeg", ".png")) and "crop" not in fname:
            ripe = extract_class(fname)
            if ripe not in VALID_CLASSES:
                print(f"[!] Ignored class: {ripe} in {fname}")
                continue
            src = os.path.join(base, fname)
            for set_name in ["original", "mixed"]:
                dst = os.path.join(output_sets[set_name], ripe, fname)
                shutil.copy2(src, dst)

    # Copy cropped images to 'crop' and 'mixed' folders
    if os.path.exists(crop_dir):
        for fname in os.listdir(crop_dir):
            if not fname.lower().endswith((".jpg", ".jpeg", ".png")):
                continue
            ripe = extract_class(fname)
            if ripe not in VALID_CLASSES:
                print(f"[!] Ignored class (crop): {ripe} in {fname}")
                continue
            src = os.path.join(crop_dir, fname)
            for set_name in ["crop", "mixed"]:
                dst = os.path.join(output_sets[set_name], ripe, fname)
                shutil.copy2(src, dst)
    else:
        print(
            "[!] 'database/crop/' directory not found. Cropped images will not be copied."
        )

    print("✅ Datasets have been organized into 'dbsets/'")


if __name__ == "__main__":
    create_dataset_structure()
