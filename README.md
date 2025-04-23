# Cuneiform Tablet Processor

A Python application built with PyQt5 for processing archaeological photographs of cuneiform tablets by removing their backgrounds, producing clean images suitable for analysis or publication.

## Features

- **Background Removal**: Uses an HSV color space-based algorithm with customizable parameters (saturation/value thresholds, morphological operations, contour filtering, feathering, and smoothing) to remove backgrounds from tablet images.
- **User Interface**: PyQt5-based GUI with real-time preview, parameter controls, and a processing log that displays only the most recently changed parameter and its associated processing step.
- **Batch Processing**: Processes multiple images in a specified folder efficiently.
- **Metadata Support**: Configurable metadata embedding for photographer, institution, and copyright information.

## Requirements

- **Python**: 3.8 or later (3.8 recommended for PyInstaller compatibility).
- **Operating System**: Windows (for .exe generation), Linux, or macOS.
- **Dependencies**:
  - `PyQt5`: For the graphical interface.
  - `opencv-python`: For image processing.
  - `numpy`: For numerical operations.
  - `pyinstaller`: For generating the executable.

## Installation

1. **Download or Clone the Project**:
   Place the project files in a directory (e.g., `cuneiform-tablet-processor`).

2. **Create a Virtual Environment** (recommended):
   ```bash
   python -m venv venv
   .\venv\Scripts\activate  # Windows
   source venv/bin/activate  # Linux/macOS
   ```

3. **Install Dependencies**:
   ```bash
   pip install PyQt5 opencv-python numpy pyinstaller
   ```

## Running the Application

1. Navigate to the project directory and activate the virtual environment (if used).
2. Run the main script:
   ```bash
   python main.py
   ```
3. The GUI allows you to:
   - Select a source folder containing tablet images (JPG, PNG, TIFF).
   - Choose an output folder for processed images.
   - Adjust background removal parameters (e.g., saturation threshold, feathering type).
   - Preview the effect on a sample image.
   - Process images in batch, with the log showing the last changed parameter (e.g., “s_threshold = 40”).

## Generating the Executable (.exe)

To create a standalone `.exe` for Windows, use PyInstaller.

### Steps

1. **Activate the Virtual Environment** (if used):
   ```bash
   .\venv\Scripts\activate  # Windows
   ```

2. **Run PyInstaller**:
   In the directory containing `main.py`, run:
   ```bash
   pyinstaller --name CuneiformTabletProcessor --windowed --onefile main.py
   ```
   - `--name`: Sets the executable name.
   - `--windowed`: Runs without a console window (GUI mode).
   - `--onefile`: Bundles into a single `.exe`.

3. **Include Assets (if applicable)**:
   If you have an `assets` folder (e.g., for icons), include it:
   ```bash
   pyinstaller --name CuneiformTabletProcessor --windowed --onefile --add-data "assets;assets" --icon="assets/app.ico" main.py
   ```
   - Use `;` for Windows, `:` for Linux/macOS in `--add-data`.
   - `--icon` sets a custom icon (optional).

4. **Locate the Executable**:
   Find the `.exe` at:
   - `dist\CuneiformTabletProcessor.exe`

5. **Test the Executable**:
   Run `CuneiformTabletProcessor.exe`. If it fails, remove `--windowed` to see errors:
   ```bash
   pyinstaller --name CuneiformTabletProcessor --onefile main.py
   ```

6. **Troubleshooting**:
   - **Missing Modules**: Add `--hidden-import` for PyQt5 issues:
     ```bash
     pyinstaller --name CuneiformTabletProcessor --windowed --onefile --hidden-import PyQt5.sip main.py
     ```
   - **Large File Size**: Expect 50–200 MB due to PyQt5 and OpenCV.
   - **Resource Paths**: If assets fail to load, consider Qt’s QResource system.

7. **Create an Installer (Optional)**:
   Use **Inno Setup** or **InstallForge** to create a setup wizard:
   - Add `dist\CuneiformTabletProcessor.exe`.
   - Configure settings (e.g., app name, version).
   - Build a `setup.exe`.

### Example Command
For a single-file `.exe` with assets and an icon:
```bash
pyinstaller --name CuneiformTabletProcessor --windowed --onefile --add-data "assets;assets" --icon="assets/app.ico" --hidden-import PyQt5.sip main.py
```

## User Guide

### Background Removal

The application uses a single HSV-based background removal method with adjustable parameters:

1. **Select Folders**:
   - **Source Folder**: Choose a folder with tablet images.
   - **Output Folder**: Specify where to save processed images.
2. **Adjust Parameters**:
   - Modify settings like saturation threshold, morphological sizes, feathering type, or smoothing.
   - The log displays the last changed parameter (e.g., “Updating preview with changed parameter: feather_type = gaussian”).
3. **Preview**:
   - Select a sample image to see the background removal effect in real-time.
4. **Process Images**:
   - Click “Remove Backgrounds” to process all images.
   - The log shows the processing step tied to the last changed parameter (e.g., “Applying gaussian feathering with amount=5”).

### Logging
- The processing log focuses on the most recently changed parameter and its related step (e.g., mask creation for `s_threshold`, feathering for `feather_amount`).
- Errors are highlighted in red with timestamps.

## Project Structure

```
tablet-processor/
├── app.py                      # Application entry point
├── gui
|   ├── __init__.py
|   ├── main_window.py          # Main UI implementation
|   ├── settings.py             # Main UI implementation
└── processing/
    ├── __init__.py
    ├── processor.py            # Core image processing pipeline
    ├── raw_processor.py        # RAW file conversion module
    └── background_remover.py   # Background removal implementations
```

## Credits and License

Developed for archaeological research to process cuneiform tablet images.

This project is licensed under the [GPL License](LICENSE), as required by PyQt5.

## Contributing

Contributions are welcome! To contribute:
1. Fork the project (if hosted).
2. Create a feature branch (`git checkout -b feature/my-feature`).
3. Commit changes (`git commit -m 'Add my feature'`).
4. Push to the branch (`git push origin feature/my-feature`).
5. Submit a pull request.