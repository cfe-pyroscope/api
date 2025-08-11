import os
from sqlmodel import Session, select
from db.db.session import engine
from db.file_scanner import scan_storage_files
from config.config import settings


def sync_dataset(dataset_name: str, storage_dir: str, table_cls):
    """
    Scans a local storage directory for dataset files and syncs new entries to the database table.

    This function:
    - Recursively scans the specified `storage_dir` for files.
    - Filters files that belong to the specified `dataset_name`.
    - Computes each file's relative path to the configured storage root.
    - Compares with existing entries in the database to identify new files.
    - Inserts new records into the provided database table class.

    Args:
        dataset_name (str): The name of the dataset to sync.
        storage_dir (str): Path to the directory where dataset files are stored.
        table_cls: The ORM table class representing the database table for the dataset.

    Returns:
        None

    Notes:
        - Only files with relative paths not already present in the database are added.
        - Files outside the configured storage root are skipped with a warning.
    """
    # Get absolute root path from config
    root = os.path.abspath(settings.STORAGE_ROOT)
    storage_dir_abs = os.path.abspath(storage_dir)

    files = scan_storage_files(storage_dir_abs)
    with Session(engine) as session:
        # Fetch existing relative filepaths in DB
        existing_paths = {row for row in session.exec(select(table_cls.filepath)).all()}

        new_entries = []
        for dataset, dt, full_path in files:
            if dataset != dataset_name:
                continue

            # Compute relative path to root storage dir
            try:
                rel_path = os.path.relpath(full_path, root)
            except ValueError:
                print(f"Warning: file {full_path} is outside storage root {root}, skipping")
                continue

            if rel_path not in existing_paths:
                new_entries.append(table_cls(dataset=dataset, datetime=dt, filepath=rel_path))

        if new_entries:
            session.add_all(new_entries)
            session.commit()
            print(f"[{dataset_name}] Inserted {len(new_entries)} new entries.")
        else:
            print(f"[{dataset_name}] No new files to add.")