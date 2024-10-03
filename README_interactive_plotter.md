### `README.md`:
# Interactive File Plotter

This Python project allows you to plot data from multiple CSV or other delimited files selected from different directories. It includes a customized file selection window with an intuitive GUI, interactive plotting with features like zoom, pan, reset, and toggling line visibility through the legend. 

## Features

- **Multiple File Selection**: Select files from different folders through a custom window.
- **Automatic Delimiter Detection**: Detects and handles single and multi-character delimiters in files (e.g., commas, tabs, spaces).
- **Interactive Plotting**: 
  - Zoom and pan functionality (`z` to zoom, `d` to pan).
  - Reset plot view (`r` key).
  - Fullscreen view (`f` key).
  - Toggle gridlines (`g` key).
  - Toggle visibility of plotted lines by clicking on corresponding legend items.
- **Custom File Selection Window**: A window with support for selecting files, with an option to select more files before finalizing.
- **Scrollbar Support**: Added horizontal and vertical scrollbars to handle longer lists of selected files.

## Requirements

- Python 3.x
- Libraries: `matplotlib`, `pandas`, `tkinter`

You can install the necessary Python packages using pip:

```bash
pip install matplotlib pandas
```

`tkinter` is part of the Python standard library, so it should be included with your Python installation.

## How to Use

1. **Run the Script**: After setting up the environment, run the script in your terminal:

    ```bash
    python interactive_plotter.py
    ```

2. **File Selection Window**: 
    - A window will appear allowing you to select files to plot. 
    - Use the "Open" button to select files, and the "Select More Files" button to continue adding files from different folders.
    - Once all files are selected, click "Finalize" to proceed, or click "Cancel" to exit.

3. **Interactive Plot**:
    - Once files are selected, an interactive plot will appear with all selected files plotted.
    - Each file's line can be toggled visible/invisible by clicking its corresponding label in the legend.

4. **Keyboard Shortcuts**:
    - `z`: Activate the zoom tool (right-click to zoom in and out).
    - `d`: Activate the pan tool (left-click to drag and move the plot).
    - `r`: Reset the plot to its original view.

## Code Breakdown

### `custom_file_selection_window()`
Creates a custom file selection window that allows the user to select multiple files across different directories. Features include:
- **Select More Files**: Adds files without reopening the window.
- **Listbox with Scrollbars**: Display the selected files with horizontal and vertical scrollbars.

### `detect_delimiter(file_path)`
Detects the delimiter in the given file by analyzing the first line. Supports:
- Comma (`,`), Tab (`\t`), Semicolon (`;`), and Spaces (` `).

### `plot_files(file_paths)`
Reads and plots the data from the selected files, and provides interactive features:
- **Legend**: Each line is labeled with the folder and file name.
- **Pick Event**: Clicking on a legend entry toggles the corresponding line's visibility. 
- **Zoom, Pan, Reset**: Controlled by keyboard shortcuts (`z`, `d`, `r`).


## Example Usage

1. Run the script and use the file selection window to choose CSV (or similar data) files.
2. The plot will display data from each file in different colors, with a legend for easy identification.
3. Interact with the plot using the provided keyboard shortcuts.


## Contributions

Feel free to contribute by creating a pull request. Contributions can include:
- Bug fixes
- Additional features (e.g., support for more file types)
- User interface improvements

## License

This project is licensed under the MIT License.
