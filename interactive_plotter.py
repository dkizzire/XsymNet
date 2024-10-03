import tkinter as tk
from tkinter import filedialog
import matplotlib.pyplot as plt
import pandas as pd
import os
import re

def custom_file_selection_window():
    """
    Custom window for selecting multiple files with the option to select more files.
    Includes 'Open', 'Cancel', and 'Select More Files' buttons.
    """
    selected_files = []
    root = tk.Tk()
    root.title("Select Files for Plotting")

    # Create and position the widgets
    instructions = tk.Label(root, text="Select files for plotting:")
    instructions.grid(row=0, column=0, padx=10, pady=10, columnspan=3)

    # Scrollbars for the Listbox
    x_scrollbar = tk.Scrollbar(root, orient=tk.HORIZONTAL)
    y_scrollbar = tk.Scrollbar(root, orient=tk.VERTICAL)

    # Listbox to display selected files
    file_listbox = tk.Listbox(root, width=100, height=10, xscrollcommand=x_scrollbar.set, yscrollcommand=y_scrollbar.set)
    file_listbox.grid(row=1, column=0, columnspan=3, padx=10, pady=10, sticky="nsew")

    # Configure scrollbars
    x_scrollbar.config(command=file_listbox.xview)
    y_scrollbar.config(command=file_listbox.yview)

    # Position scrollbars
    x_scrollbar.grid(row=2, column=0, columnspan=3, sticky="ew")
    y_scrollbar.grid(row=1, column=3, sticky="ns")

    def open_files():
        # Open file explorer to select files
        files = filedialog.askopenfilenames(title="Select Files")
        if files:
            selected_files.extend(files)
            # Update the listbox with the newly selected files
            file_listbox.delete(0, tk.END)
            for file in selected_files:
                file_listbox.insert(tk.END, file)

    def finalize_selection():
        # Close the window and return the selected files
        root.quit()

    def select_more_files():
        open_files()

    def cancel_selection():
        selected_files.clear()  # Clear any selected files if the user cancels
        root.quit()

    # Open button to select files initially
    open_button = tk.Button(root, text="Open", command=open_files)
    open_button.grid(row=3, column=0, padx=10, pady=10)

    # Button to allow the user to select more files
    more_button = tk.Button(root, text="Select More Files", command=select_more_files)
    more_button.grid(row=3, column=1, padx=10, pady=10)

    # Finalize selection button (acts like 'OK')
    finalize_button = tk.Button(root, text="Finalize", command=finalize_selection)
    finalize_button.grid(row=3, column=2, padx=10, pady=10)

    # Cancel button to cancel file selection
    cancel_button = tk.Button(root, text="Cancel", command=cancel_selection)
    cancel_button.grid(row=3, column=3, padx=10, pady=10)

    root.mainloop()  # Start the window

    root.destroy()  # Destroy the window after selection is done
    return selected_files

def detect_delimiter(file_path):
    """
    Automatically detects the delimiter used in a file by analyzing the first line.
    """
    with open(file_path, 'r') as file:
        first_line = file.readline().strip()
        delimiters = [',', '\t', ';', ' ']
        delimiter_counts = {delim: len(re.findall(re.escape(delim), first_line)) for delim in delimiters}
        return max(delimiter_counts, key=delimiter_counts.get)

def plot_files(file_paths):
    """
    Reads and plots the data from selected files, even if they are from different folders.
    Adds interactive features for zooming, panning, editing line properties, and toggling line visibility.
    """
    fig, ax = plt.subplots()

    lines = []  # Store line objects
    labels = []  # Store labels for legend

    for file_path in file_paths:
        try:
            delimiter = detect_delimiter(file_path)
            
            if delimiter not in [',', '\t', ';', ' ']:
                print(f"Unsupported delimiter detected in {file_path}. Skipping this file.")
                continue

            data = pd.read_csv(file_path, sep=delimiter, skipinitialspace=True)

            if data.shape[1] > 2:
                print(f"File {file_path} contains more than 2 columns. Select columns to plot:")
                print(data.columns.tolist())
                col_x = int(input(f"Choose X column (0 to {data.shape[1] - 1}): "))
                col_y = int(input(f"Choose Y column (0 to {data.shape[1] - 1}): "))
                x = data.iloc[:, col_x]
                y = data.iloc[:, col_y]
            else:
                x = data.iloc[:, 0]
                y = data.iloc[:, 1]

            folder_name, file_name = os.path.split(file_path)
            line, = ax.plot(x, y, label=f"{os.path.basename(folder_name)}/{file_name}")
            lines.append(line)
            labels.append(f"{os.path.basename(folder_name)}/{file_name}")

        except Exception as e:
            print(f"Error reading {file_path}: {e}")

    # Create legend and make lines pickable
    legend = ax.legend(loc='upper right')

    def on_pick(event):
        """
        Toggles the visibility of the corresponding line when a legend entry is clicked.
        """
        legline = event.artist
        index = legend.get_lines().index(legline)
        line = lines[index]
        
        # Toggle visibility
        visible = not line.get_visible()
        line.set_visible(visible)
        
        # Change the legend line transparency
        legline.set_alpha(1.0 if visible else 0.2)
        
        fig.canvas.draw()

    # Set pickable state for each legend line
    for legline in legend.get_lines():
        legline.set_picker(True)

    fig.canvas.mpl_connect('pick_event', on_pick)

    original_xlim = ax.get_xlim()
    original_ylim = ax.get_ylim()

    def on_key(event):
        if event.key == 'z':
            plt.gca().figure.canvas.toolbar.zoom()

        elif event.key == 'd':
            plt.gca().figure.canvas.toolbar.pan()

        elif event.key == 'r':
            ax.set_xlim(original_xlim)
            ax.set_ylim(original_ylim)
            fig.canvas.draw()

    fig.canvas.mpl_connect('key_press_event', on_key)
    plt.show()

# Example usage
selected_files = custom_file_selection_window()
if selected_files:
    plot_files(selected_files)
else:
    print("No files selected.")
