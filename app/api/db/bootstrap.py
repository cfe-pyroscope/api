import os
from sqlmodel import Session, select
from app.api.db.session import engine
from app.services.file_scanner import scan_storage_files
from app.api.config import settings


def sync_dataset(dataset_name: str, storage_dir: str, table_cls):
    """
    Synchronize NetCDF files in a given storage directory with the database.

    This function scans the specified storage directory for `.nc` files,
    parses metadata (dataset name and timestamp) from the filenames,
    and inserts new records into the database table associated with `table_cls`
    if they are not already present.

    Only files matching the specified `dataset_name` are processed.
    File paths are stored relative to the configured `STORAGE_ROOT`.

    Args:
        dataset_name (str): The expected dataset name to filter files (e.g., "fopi" or "pof").
        storage_dir (str): Path to the directory containing `.nc` files.
        table_cls: The SQLModel class corresponding to the database table to insert into (e.g., `Fopi` or `Pof`).

    Side Effects:
        - Inserts new entries into the database if matching files are found.
        - Prints status messages about inserted or skipped files.

    Warnings:
        - Files located outside the `STORAGE_ROOT` path are ignored and logged with a warning.

    Raises:
        None directly, but errors in file scanning, parsing, or DB session could raise exceptions implicitly.
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
