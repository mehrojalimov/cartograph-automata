#!/usr/bin/env python3
"""
CV Training Data Collection Script
This script efficiently copies images with corresponding JSON labels from multiple source directories
to a single destination directory for computer vision training. It validates image formats,
avoids duplicates, and can generate detailed summaries of the training data.

Author: Rokawoo
"""
import os
import sys
import shutil
import argparse
from pathlib import Path
from concurrent.futures import ThreadPoolExecutor
import hashlib
from PIL import Image
from collections import Counter

# Get the directory where the script is located
SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))

# Source data parent directory
SOURCE_DATA_DIR = os.path.join(SCRIPT_DIR, "captured_cv_training_data")

# Set default destination to be in the same directory as the script
DEFAULT_DESTINATION = os.path.join(SCRIPT_DIR, "labeled_cv_training_data")

DEFAULT_THREADS = 4
SUPPORTED_FORMATS = {'JPEG', 'PNG', 'GIF', 'BMP'}

def get_source_folders():
    """Auto-detect source folders from the captured_cv_training_data directory."""
    source_parent = Path(SOURCE_DATA_DIR)
    
    if not source_parent.exists():
        print(f"Warning: Source parent directory '{SOURCE_DATA_DIR}' does not exist.")
        print(f"Creating it now. Please add your data folders inside it.")
        source_parent.mkdir(parents=True, exist_ok=True)
        return []
    
    # Get all subdirectories in the source parent directory
    source_folders = [str(d) for d in source_parent.iterdir() if d.is_dir()]
    
    if not source_folders:
        print(f"Warning: No subdirectories found in '{SOURCE_DATA_DIR}'.")
        print(f"Please add folders containing your source data in that directory.")
    else:
        print(f"Found {len(source_folders)} source data directories:")
        for folder in source_folders:
            print(f"  - {folder}")
    
    return source_folders

def get_file_hash(filepath):
    """Generate a hash of the file contents for comparison."""
    BUF_SIZE = 65536  # 64kb chunks
    sha256 = hashlib.sha256()
    
    with open(filepath, 'rb') as f:
        while True:
            data = f.read(BUF_SIZE)
            if not data:
                break
            sha256.update(data)
    
    return sha256.hexdigest()

def is_valid_image(image_path):
    """Check if the image is in a supported format."""
    try:
        with Image.open(image_path) as img:
            img_format = img.format
            if img_format is None or img_format.upper() not in SUPPORTED_FORMATS:
                return False, img_format
            return True, img_format
    except Exception as e:
        return False, str(e)

def analyze_directory(directory_path):
    """Analyze a directory and return statistics about the files."""
    path = Path(directory_path)
    
    if not path.exists() or not path.is_dir():
        print(f"Error: {directory_path} does not exist or is not a directory.")
        return None
    
    stats = {
        'total_files': 0,
        'image_files': 0,
        'json_files': 0,
        'image_json_pairs': 0,
        'orphaned_images': 0,
        'orphaned_jsons': 0,
        'invalid_images': 0,
        'image_formats': Counter(),
        'total_size_mb': 0,
        'invalid_format_details': []
    }
    
    # Get all files and categorize them
    all_files = list(path.glob('*'))
    stats['total_files'] = len(all_files)
    
    image_stems = set()
    json_stems = set()
    
    # First pass - categorize files
    for file in all_files:
        file_size_mb = file.stat().st_size / (1024 * 1024)
        stats['total_size_mb'] += file_size_mb
        
        if file.suffix.lower() in ['.png', '.jpg', '.jpeg', '.bmp', '.gif']:
            stats['image_files'] += 1
            image_stems.add(file.stem)
            
            # Check if image is valid
            is_valid, format_info = is_valid_image(file)
            if not is_valid:
                stats['invalid_images'] += 1
                stats['invalid_format_details'].append((file.name, format_info))
            else:
                stats['image_formats'][format_info] += 1
                
        elif file.suffix.lower() == '.json':
            stats['json_files'] += 1
            json_stems.add(file.stem)
    
    # Find pairs and orphans
    paired_stems = image_stems.intersection(json_stems)
    stats['image_json_pairs'] = len(paired_stems)
    stats['orphaned_images'] = len(image_stems - paired_stems)
    stats['orphaned_jsons'] = len(json_stems - paired_stems)
    
    return stats

def generate_summary(directory_path):
    """Generate and print a summary of the directory contents."""
    print(f"\n===== SUMMARY FOR {directory_path} =====")
    
    stats = analyze_directory(directory_path)
    if not stats:
        return
    
    # Print general statistics
    print(f"\nGeneral Statistics:")
    print(f"  Total files:           {stats['total_files']}")
    print(f"  Total size:            {stats['total_size_mb']:.2f} MB")
    print(f"  Image files:           {stats['image_files']}")
    print(f"  JSON label files:      {stats['json_files']}")
    print(f"  Image-JSON pairs:      {stats['image_json_pairs']}")
    
    # Print orphaned files information
    if stats['orphaned_images'] > 0 or stats['orphaned_jsons'] > 0:
        print(f"\nOrphaned Files (no matching pair):")
        print(f"  Orphaned images:       {stats['orphaned_images']}")
        print(f"  Orphaned JSON files:   {stats['orphaned_jsons']}")
    
    # Print image format information
    print(f"\nImage Format Statistics:")
    for format_name, count in stats['image_formats'].items():
        print(f"  {format_name}: {count}")
    
    # Print invalid image information
    if stats['invalid_images'] > 0:
        print(f"\nInvalid Images: {stats['invalid_images']}")
        for i, (filename, reason) in enumerate(stats['invalid_format_details'][:10]):
            print(f"  {i+1}. {filename}: {reason}")
        
        if len(stats['invalid_format_details']) > 10:
            print(f"  ... and {len(stats['invalid_format_details'])-10} more")
    
    # Print validation result
    if stats['invalid_images'] == 0:
        print(f"\n✅ All images are valid and in supported formats.")
    else:
        print(f"\n❌ Found {stats['invalid_images']} invalid images that may cause issues.")
    
    if stats['orphaned_images'] == 0 and stats['orphaned_jsons'] == 0:
        print(f"✅ All files have matching pairs (images have JSON labels and vice versa).")
    else:
        print(f"⚠️ Found orphaned files that don't have matching pairs.")

def process_directory(source_dir, dest_dir, copied_files, dry_run=False):
    """Process a single directory, finding matching image and JSON pairs."""
    source_path = Path(source_dir)
    if not source_path.exists() or not source_path.is_dir():
        print(f"Warning: Source directory {source_dir} does not exist or is not a directory")
        return 0
    
    print(f"Processing directory: {source_dir}")
    
    # Get all files in the directory
    all_files = list(source_path.glob('*'))
    
    # Create dictionaries to store filenames without extensions
    image_files = {}
    json_files = {}
    
    # Categorize files by extension
    for file in all_files:
        stem = file.stem
        if file.suffix.lower() in ['.png', '.jpg', '.jpeg', '.bmp', '.gif']:
            image_files[stem] = file
        elif file.suffix.lower() == '.json':
            json_files[stem] = file
    
    # Find files that have both image and JSON
    pairs_found = 0
    skipped_invalid_formats = 0
    
    for stem in set(image_files.keys()) & set(json_files.keys()):
        image_file = image_files[stem]
        json_file = json_files[stem]
        
        # Validate image format
        is_valid, format_info = is_valid_image(image_file)
        if not is_valid:
            print(f"Skipping unsupported image format: {image_file.name} ({format_info})")
            skipped_invalid_formats += 1
            continue
        
        # Create destination paths
        dest_image = Path(dest_dir) / image_file.name
        dest_json = Path(dest_dir) / json_file.name
        
        # Check if we need to copy the files
        image_needs_copy = True
        json_needs_copy = True
        
        # Check if destination files already exist
        if dest_image.exists():
            if not overwrite:
                # Check if the files are identical by hash
                if get_file_hash(image_file) == get_file_hash(dest_image):
                    image_needs_copy = False
                    print(f"Skipping identical image: {image_file.name}")
                else:
                    print(f"File exists but different content: {image_file.name}")
        
        if dest_json.exists():
            if not overwrite:
                if get_file_hash(json_file) == get_file_hash(dest_json):
                    json_needs_copy = False
                    print(f"Skipping identical JSON: {json_file.name}")
                else:
                    print(f"File exists but different content: {json_file.name}")
        
        # Copy files if needed
        if image_needs_copy or json_needs_copy:
            if dry_run:
                print(f"Would copy: {image_file.name} and {json_file.name}")
                pairs_found += 1
                continue
                
            try:
                if image_needs_copy:
                    shutil.copy2(image_file, dest_image)
                    copied_files.add(image_file.name)
                    print(f"Copied image: {image_file.name}")
                
                if json_needs_copy:
                    shutil.copy2(json_file, dest_json)
                    copied_files.add(json_file.name)
                    print(f"Copied JSON: {json_file.name}")
                
                pairs_found += 1
            except Exception as e:
                print(f"Error copying {image_file.name} or {json_file.name}: {str(e)}")
    
    print(f"Directory {source_dir}: Found {pairs_found} valid pairs, skipped {skipped_invalid_formats} invalid images")
    return pairs_found

def main():
    parser = argparse.ArgumentParser(description='Copy images with corresponding JSON labels to a destination folder.')
    parser.add_argument('--destination', '-d', default=DEFAULT_DESTINATION, 
                        help=f'Destination directory for copied files (default in script directory)')
    parser.add_argument('--threads', '-t', type=int, default=DEFAULT_THREADS, 
                        help=f'Number of threads to use for processing (default: {DEFAULT_THREADS})')
    parser.add_argument('--overwrite', '-o', action='store_true', help='Overwrite existing files in destination')
    parser.add_argument('--dry-run', action='store_true', help='Show what would be copied without actually copying')
    parser.add_argument('--summary', action='store_true',
                        help='Generate a summary of the destination directory without copying files')
    
    args = parser.parse_args()
    
    # Print script location info
    print(f"Script is located at: {SCRIPT_DIR}")
    print(f"Looking for source folders in: {SOURCE_DATA_DIR}")
    print(f"Output folder will be: {args.destination}\n---")
    
    # If summary mode is enabled, just generate the summary and exit
    if args.summary:
        generate_summary(args.destination)
        return 0
    
    # Auto-detect source folders from the captured_cv_training_data directory
    source_folders = get_source_folders()
    
    if not source_folders:
        print("\nNo source folders found. Please add data directories inside:")
        print(f"  {SOURCE_DATA_DIR}")
        print("\nRun the script again after adding source data folders.")
        return 1
    
    # Print summary of what we're going to do
    print(f"Using {args.threads} threads for processing")
    
    # Create destination directory if it doesn't exist
    dest_dir = Path(args.destination)
    if not args.dry_run:
        dest_dir.mkdir(parents=True, exist_ok=True)
        print(f"Created destination directory: {dest_dir}")
    
    # Set global overwrite flag
    global overwrite
    overwrite = args.overwrite
    
    # Track already copied files
    copied_files = set()
    
    # Process directories in parallel
    with ThreadPoolExecutor(max_workers=args.threads) as executor:
        futures = [executor.submit(process_directory, source, args.destination, copied_files, args.dry_run) 
                  for source in source_folders]
        
        # Collect results
        total_pairs = 0
        for future in futures:
            total_pairs += future.result()
    
    print(f"\nSummary:")
    print(f"{'Would copy' if args.dry_run else 'Copied'} {total_pairs} image-JSON pairs")
    print(f"Total unique files {'that would be transferred' if args.dry_run else 'transferred'}: {len(copied_files)}")
    
    # Generate a full summary after copying
    if not args.dry_run and total_pairs > 0:
        generate_summary(args.destination)
    
    return 0

if __name__ == "__main__":
    sys.exit(main())