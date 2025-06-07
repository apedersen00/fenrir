#!/usr/bin/env python3
"""
Copy Memory Dump Files from Vivado Simulation to Project Directory
Updated with option to delete source files after copying
"""

import shutil
import os
from pathlib import Path
import glob

def copy_memory_dumps(delete_after_copy=False):
    """Copy all memory dump files from Vivado sim directory to project directory"""
    
    # Source and destination paths
    source_dir = Path(r"E:\rtsprojects\sv_convolutions\sv_convolutions.sim\verification\behav\xsim")
    dest_dir = Path(r"C:\Users\alext\source\fenrir\snntorch\convolution\sv_conv\python\mem_dumps")
    
    print("=== Memory Dump File Copy Script ===")
    print(f"Source: {source_dir}")
    print(f"Destination: {dest_dir}")
    if delete_after_copy:
        print("⚠️  DELETE MODE: Source files will be deleted after copying!")
    print()
    
    # Check if source directory exists
    if not source_dir.exists():
        print(f"❌ Error: Source directory does not exist!")
        print(f"    {source_dir}")
        return False
    
    # Create destination directory if it doesn't exist
    dest_dir.mkdir(parents=True, exist_ok=True)
    print(f"✓ Destination directory ready: {dest_dir}")
    
    # Find all memory dump files
    memory_dump_patterns = [
        "memory_dump*.csv",
        "memory_dumps.csv",
        "*memory*.csv"
    ]
    
    files_found = []
    for pattern in memory_dump_patterns:
        pattern_files = list(source_dir.glob(pattern))
        files_found.extend(pattern_files)
    
    # Remove duplicates while preserving order
    files_found = list(dict.fromkeys(files_found))
    
    if not files_found:
        print("❌ No memory dump files found!")
        print("   Looking for files matching: memory_dump*.csv, memory_dumps.csv")
        print("   Available files in source directory:")
        
        all_files = list(source_dir.glob("*.csv"))
        if all_files:
            for file in sorted(all_files):
                print(f"     {file.name}")
        else:
            print("     No CSV files found")
        return False
    
    # Copy files and track successful copies for deletion
    print(f"\n📁 Found {len(files_found)} memory dump files:")
    copied_count = 0
    files_to_delete = []  # Track successfully copied files
    
    for src_file in sorted(files_found):
        try:
            dest_file = dest_dir / src_file.name
            
            # Copy file
            shutil.copy2(src_file, dest_file)
            
            # Verify the copy was successful by checking file exists and size matches
            if dest_file.exists() and dest_file.stat().st_size == src_file.stat().st_size:
                size_kb = src_file.stat().st_size / 1024
                print(f"  ✓ {src_file.name} ({size_kb:.1f} KB)")
                copied_count += 1
                
                # Add to deletion list only if copy was verified successful
                if delete_after_copy:
                    files_to_delete.append(src_file)
            else:
                print(f"  ❌ Copy verification failed for {src_file.name}")
            
        except Exception as e:
            print(f"  ❌ Failed to copy {src_file.name}: {e}")
    
    print(f"\n🎉 Successfully copied {copied_count}/{len(files_found)} files!")
    
    # Delete source files if requested and copies were successful
    if delete_after_copy and files_to_delete:
        print(f"\n🗑️  Deleting {len(files_to_delete)} source files...")
        deleted_count = 0
        
        for src_file in files_to_delete:
            try:
                src_file.unlink()
                print(f"  ✓ Deleted {src_file.name}")
                deleted_count += 1
            except Exception as e:
                print(f"  ❌ Failed to delete {src_file.name}: {e}")
        
        print(f"🗑️  Successfully deleted {deleted_count}/{len(files_to_delete)} source files!")
        
        if deleted_count != len(files_to_delete):
            print("⚠️  Some source files could not be deleted - check permissions")
    
    # Show what's now in the destination directory
    print(f"\n📂 Files now in destination directory:")
    dest_files = sorted(dest_dir.glob("*.csv"))
    
    for file in dest_files:
        size_kb = file.stat().st_size / 1024
        print(f"   {file.name} ({size_kb:.1f} KB)")
    
    return copied_count > 0

def clean_destination():
    """Clean old memory dump files from destination directory"""
    
    dest_dir = Path(r"C:\Users\alext\source\fenrir\snntorch\convolution\sv_conv\python\mem_dumps")
    
    if not dest_dir.exists():
        print("Destination directory doesn't exist - nothing to clean")
        return
    
    # Find old memory dump files
    old_files = list(dest_dir.glob("memory_dump*.csv")) + list(dest_dir.glob("memory_dumps.csv"))
    
    if not old_files:
        print("No old memory dump files to clean")
        return
    
    print(f"\n🧹 Cleaning {len(old_files)} old memory dump files...")
    
    for file in old_files:
        try:
            file.unlink()
            print(f"  ✓ Removed {file.name}")
        except Exception as e:
            print(f"  ❌ Failed to remove {file.name}: {e}")

def clean_source_dumps():
    """Clean memory dump files from source directory (Vivado sim directory)"""
    
    source_dir = Path(r"E:\rtsprojects\sv_convolutions\sv_convolutions.sim\verification\behav\xsim")
    
    if not source_dir.exists():
        print("Source directory doesn't exist - nothing to clean")
        return
    
    # Find memory dump files in source
    memory_dump_patterns = [
        "memory_dump*.csv",
        "memory_dumps.csv",
        "*memory*.csv"
    ]
    
    files_found = []
    for pattern in memory_dump_patterns:
        pattern_files = list(source_dir.glob(pattern))
        files_found.extend(pattern_files)
    
    # Remove duplicates
    files_found = list(dict.fromkeys(files_found))
    
    if not files_found:
        print("No memory dump files found in source directory")
        return
    
    print(f"\n🧹 Cleaning {len(files_found)} memory dump files from source...")
    
    for file in files_found:
        try:
            file.unlink()
            print(f"  ✓ Removed {file.name}")
        except Exception as e:
            print(f"  ❌ Failed to remove {file.name}: {e}")

def main():
    """Main function with options"""
    
    import argparse
    
    parser = argparse.ArgumentParser(description='Copy memory dump files from Vivado simulation')
    parser.add_argument('--clean', action='store_true', 
                       help='Clean old memory dump files from destination before copying')
    parser.add_argument('--clean-source', action='store_true',
                       help='Clean memory dump files from source directory (no copying)')
    parser.add_argument('--delete-after-copy', action='store_true',
                       help='Delete source files after successful copying')
    parser.add_argument('--list-only', action='store_true',
                       help='Only list files, do not copy')
    
    args = parser.parse_args()
    
    # Clean source files only (no copying)
    if args.clean_source:
        clean_source_dumps()
        return
    
    # Clean destination files if requested
    if args.clean:
        clean_destination()
    
    if args.list_only:
        # Just list what files would be copied
        source_dir = Path(r"E:\rtsprojects\sv_convolutions\sv_convolutions.sim\verification\behav\xsim")
        
        print("=== Files that would be copied ===")
        print(f"From: {source_dir}")
        
        if not source_dir.exists():
            print("❌ Source directory does not exist!")
            return
        
        files_found = []
        for pattern in ["memory_dump*.csv", "memory_dumps.csv", "*memory*.csv"]:
            files_found.extend(source_dir.glob(pattern))
        
        files_found = list(dict.fromkeys(files_found))
        
        if files_found:
            for file in sorted(files_found):
                size_kb = file.stat().st_size / 1024
                print(f"  {file.name} ({size_kb:.1f} KB)")
        else:
            print("  No memory dump files found")
    else:
        # Actually copy the files (and optionally delete source)
        success = copy_memory_dumps(delete_after_copy=args.delete_after_copy)
        
        if success:
            print("\n✅ Copy operation completed successfully!")
            if args.delete_after_copy:
                print("🗑️  Source files have been cleaned up!")
            print("\nNext steps:")
            print("  1. cd mem_dumps")
            print("  2. python3 ../verification.py memory_dumps.csv -e ../test_events.txt")
        else:
            print("\n❌ Copy operation failed!")

if __name__ == "__main__":
    main()