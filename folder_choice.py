import tkinter as tk
from tkinter import filedialog
import glob
import os

def select_folder_and_glob_files():
    # Create a root window and hide it
    root = tk.Tk()
    root.withdraw()
    
    # Open file explorer dialog to select folder
    folder_selected = filedialog.askdirectory(title="Select Folder")
    
    if not folder_selected:
        print("No folder selected.")
        return None
    
    # Glob all subdirectories and files within the selected folder
    all_files = glob.glob(os.path.join(folder_selected, '**', '*'), recursive=True)
    
    return folder_selected, all_files

# Example usage
selected_folder, files = select_folder_and_glob_files()
if selected_folder:
    print(f"Selected folder: {selected_folder}")
    print(f"Files in folder and subfolders:")
    for file in files:
        print(file)

