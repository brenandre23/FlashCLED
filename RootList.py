import os

def list_directory_structure(root_dir):
    """Recursively lists all files and directories starting from the root_dir."""
    for root, dirs, files in os.walk(root_dir):
        level = root.replace(root_dir, '').count(os.sep)
        indent = ' ' * 4 * level
        print(f"{indent}[DIR] {os.path.basename(root)}/")
        
        # Print all files in the current directory
        sub_indent = ' ' * 4 * (level + 1)
        for file in files:
            print(f"{sub_indent}[FILE] {file}")

# List the contents of the specified directory
list_directory_structure(r"C:\Users\Brenan\Desktop\Thesis\Scratch")
