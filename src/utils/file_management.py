import shutil

def move_file_up_to_new_dir(file_path, new_dir_name, new_file_name):
    # Get the absolute path of the file (resolving relative paths)
    abs_file_path = os.path.abspath(file_path)
    # Get the parent directory (one level up from file's current directory)
    parent_dir = os.path.dirname(os.path.dirname(abs_file_path))
    # Construct the path for the new directory
    new_dir_path = os.path.join(parent_dir, new_dir_name)
    # Create the new directory if it doesn't exist
    os.makedirs(new_dir_path, exist_ok=True)
    # Construct the full path for the renamed file in the new directory
    new_file_path = os.path.join(new_dir_path, new_file_name)
    print(abs_file_path, new_file_path)
    # Move (and rename) the file to the new location
    shutil.move(abs_file_path, new_file_path)
    return new_file_path