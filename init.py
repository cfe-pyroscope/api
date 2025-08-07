import os


def create_project_structure(base_path, folders):
    """Creates the folder structure if it doesn't exist.

    Args:
        base_path (str): The base path where the folders will be created.
        folders (list): A list of folder paths to create within the base path.

    Returns:
        None: This function doesn't return anything but prints status messages.

    Raises:
        Exception: If an error occurs while creating a folder, the exception is caught
            and a message is printed.
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
    "app/",
    "app/api/",
    "app/api/crud/"
    "app/api/db/"
    "app/api/models/"
    "app/api/routes/"
    "app/api/utils/"
    "app/sevices",
    "data/",
    "data/nc/",
    "data/nc/fopi/",
    "data/nc/pof/",
    "data/zarr/",
    "data/zarr/fopi/",
    "data/zarr/pof/",
    "notes",
]

create_project_structure(base_path, folders)
