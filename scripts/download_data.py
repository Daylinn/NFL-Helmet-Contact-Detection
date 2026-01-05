"""Download NFL Impact Detection dataset from Kaggle."""

import argparse
import os
import subprocess
import sys
import zipfile
from pathlib import Path


def check_kaggle_auth() -> bool:
    """Check if Kaggle credentials are configured.

    Returns:
        True if credentials found, False otherwise
    """
    # Check environment variables
    if "KAGGLE_USERNAME" in os.environ and "KAGGLE_KEY" in os.environ:
        print("✓ Found Kaggle credentials in environment variables")
        return True

    # Check kaggle.json file
    kaggle_json = Path.home() / ".kaggle" / "kaggle.json"
    if kaggle_json.exists():
        print(f"✓ Found Kaggle credentials at {kaggle_json}")
        return True

    print("✗ Kaggle credentials not found!")
    print("\nPlease set up Kaggle authentication:")
    print("  Option 1: Set environment variables")
    print("    export KAGGLE_USERNAME=your_username")
    print("    export KAGGLE_KEY=your_api_key")
    print("\n  Option 2: Create ~/.kaggle/kaggle.json")
    print('    {"username":"your_username","key":"your_api_key"}')
    print("\nGet your API key from: https://www.kaggle.com/settings/account")
    return False


def download_competition_data(competition: str, output_dir: Path) -> bool:
    """Download competition data using Kaggle CLI.

    Args:
        competition: Kaggle competition name
        output_dir: Directory to save downloaded data

    Returns:
        True if successful, False otherwise
    """
    output_dir.mkdir(parents=True, exist_ok=True)

    print(f"\nDownloading {competition} dataset...")
    print(f"Output directory: {output_dir}")

    try:
        # Download using kaggle CLI
        cmd = [
            "kaggle",
            "competitions",
            "download",
            "-c",
            competition,
            "-p",
            str(output_dir),
        ]

        result = subprocess.run(cmd, capture_output=True, text=True, check=False)

        if result.returncode != 0:
            print(f"✗ Download failed: {result.stderr}")
            return False

        print("✓ Download completed successfully")
        return True

    except FileNotFoundError:
        print("✗ Kaggle CLI not found. Please install: pip install kaggle")
        return False
    except Exception as e:
        print(f"✗ Download failed: {e}")
        return False


def unzip_data(zip_path: Path, output_dir: Path) -> bool:
    """Unzip downloaded data.

    Args:
        zip_path: Path to zip file
        output_dir: Directory to extract to

    Returns:
        True if successful, False otherwise
    """
    if not zip_path.exists():
        print(f"✗ Zip file not found: {zip_path}")
        return False

    print(f"\nExtracting {zip_path.name}...")

    try:
        with zipfile.ZipFile(zip_path, "r") as zip_ref:
            zip_ref.extractall(output_dir)

        print(f"✓ Extracted to {output_dir}")

        # Clean up zip file
        zip_path.unlink()
        print(f"✓ Removed {zip_path.name}")

        return True

    except Exception as e:
        print(f"✗ Extraction failed: {e}")
        return False


def discover_files(data_dir: Path) -> dict:
    """Auto-discover expected files in the downloaded data.

    Args:
        data_dir: Directory containing extracted data

    Returns:
        Dictionary mapping file types to paths
    """
    print("\nDiscovering files...")

    discovered = {
        "csvs": [],
        "videos": [],
        "labels": [],
        "tracking": [],
    }

    # Find all files
    for file_path in data_dir.rglob("*"):
        if not file_path.is_file():
            continue

        name_lower = file_path.name.lower()

        # CSV files
        if file_path.suffix == ".csv":
            discovered["csvs"].append(file_path)

            # Categorize CSVs
            if "label" in name_lower or "video" in name_lower:
                discovered["labels"].append(file_path)
            elif "track" in name_lower or "player" in name_lower:
                discovered["tracking"].append(file_path)

        # Video files
        elif file_path.suffix in [".mp4", ".avi", ".mov"]:
            discovered["videos"].append(file_path)

    # Print summary
    print("\nFound files:")
    print(f"  CSVs: {len(discovered['csvs'])}")
    if discovered["labels"]:
        print(f"    Labels: {len(discovered['labels'])}")
        for f in discovered["labels"]:
            print(f"      - {f.relative_to(data_dir)}")
    if discovered["tracking"]:
        print(f"    Tracking: {len(discovered['tracking'])}")
        for f in discovered["tracking"]:
            print(f"      - {f.relative_to(data_dir)}")

    print(f"  Videos: {len(discovered['videos'])}")
    if discovered["videos"]:
        # Show first few videos
        for f in list(discovered["videos"])[:5]:
            print(f"    - {f.relative_to(data_dir)}")
        if len(discovered["videos"]) > 5:
            print(f"    ... and {len(discovered['videos']) - 5} more")

    return discovered


def main():
    parser = argparse.ArgumentParser(description="Download NFL Impact Detection dataset")
    parser.add_argument(
        "--competition",
        default="nfl-impact-detection",
        help="Kaggle competition name",
    )
    parser.add_argument(
        "--output-dir",
        default="data/raw",
        help="Output directory for downloaded data",
    )
    parser.add_argument(
        "--no-unzip",
        action="store_true",
        help="Skip automatic unzipping",
    )
    parser.add_argument(
        "--force",
        action="store_true",
        help="Force re-download even if data exists",
    )

    args = parser.parse_args()

    output_dir = Path(args.output_dir)

    # Check if data already exists
    if not args.force and output_dir.exists() and list(output_dir.iterdir()):
        print(f"Data directory {output_dir} already exists and is not empty")
        print("Use --force to re-download")

        # Still try to discover files
        discovered = discover_files(output_dir)
        if discovered["csvs"] or discovered["videos"]:
            print("\n✓ Data appears to be present")
            return 0
        else:
            print("\n⚠ Data directory exists but appears empty, proceeding with download")

    # Check Kaggle authentication
    if not check_kaggle_auth():
        return 1

    # Download data
    if not download_competition_data(args.competition, output_dir):
        return 1

    # Find zip files
    zip_files = list(output_dir.glob("*.zip"))

    if not zip_files and not args.no_unzip:
        print("\n⚠ No zip files found to extract")
    elif not args.no_unzip:
        # Unzip all found zip files
        for zip_file in zip_files:
            if not unzip_data(zip_file, output_dir):
                print(f"⚠ Failed to extract {zip_file.name}")

    # Discover files
    discovered = discover_files(output_dir)

    # Validate we have expected data
    if not discovered["labels"]:
        print("\n⚠ Warning: No label CSV files found")
        print("Expected files with 'label' or 'video' in the name")

    if not discovered["videos"]:
        print("\n⚠ Warning: No video files found")
        print("Expected .mp4, .avi, or .mov files")

    if discovered["labels"] and discovered["videos"]:
        print("\n✓ Dataset download complete!")
    else:
        print("\n⚠ Download completed but some expected files may be missing")
        print("Please check the data directory manually")

    return 0


if __name__ == "__main__":
    sys.exit(main())
