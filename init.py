import os

def create_project_structure(base_path, folders):
    """
    Creates the folder structure if it doesn't exist.

    Parameters:
    base_path (str): The base path where the folders will be created.
    folders (list): A list of folder paths to create within the base path.
    """
    for folder in folders:
        folder_path = os.path.join(base_path, folder)
        try:
            os.makedirs(folder_path, exist_ok=True)
            print(f"Folder '{folder_path}' created successfully.")
        except Exception as e:
            print(f"An error occurred while creating the folder '{folder_path}': {e}")

# Example usage
base_path = "."
folders = [
    "alembic",
    "data/",
    "data/nc/",
    "data/zarr",
    "modules",
    "routes"
]

create_project_structure(base_path, folders)