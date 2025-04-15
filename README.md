# Cuneiform Tablet Processor

A comprehensive Python application for processing archaeological photographs of cuneiform tablets from RAW files to publication-ready images with professional layouts similar to museum standards.

## Features

- **RAW Processing**: Convert camera RAW files to high-quality TIFF with customizable settings
- **Background Removal**: 10 specialized algorithms optimized for archaeological artifacts
- **Tablet Composition**: Automatically arrange multiple views in standardized layouts
- **Metadata Handling**: Embed comprehensive IPTC/EXIF metadata for proper attribution
- **Scale Detection**: Automatically detect color scale in images for consistent sizing
- **Publication Output**: Generate high-quality TIFF and JPEG outputs with optional logo insertion

## Requirements

- Python 3.6+
- PyQt5
- OpenCV 4.x
- PIL/Pillow
- NumPy
- Rawpy (for RAW file processing)
- Piexif (for metadata handling)

## Installation

```bash
# Clone the repository
git clone https://github.com/yourusername/cuneiform-tablet-processor.git
cd cuneiform-tablet-processor

# Install dependencies
pip install -r requirements.txt

# Run the application
python main.py
```

## User Guide

### Processing RAW Files

The RAW Processing tab allows conversion of camera RAW files to TIFF format:

- Select source folder containing RAW files
- Choose output destination
- Configure white balance, bit depth, and resolution
- Select TIFF compression method
- Process batch with a single click

### Background Removal

The application includes 10 different background removal algorithms:

1. **Gaussian + Otsu** - For noisy backgrounds
2. **Simple Thresholding** - Fast processing for white backgrounds
3. **HSV Space** - Better for reflective surfaces
4. **Black Background** - Optimized for dark studio photography
5. **Edge Detection** - For tablets with clear boundaries
6. **Color Clustering** - For complex backgrounds with varying colors
7. **GrabCut - Auto** - Enhanced algorithm with tablet-specific optimizations
8. **Neural Network** - High-quality segmentation using deep learning
9. **Otsu Watershed** - Combines thresholding with contour refinement
10. **U-Net/DeepLabV3** - Advanced deep learning models

Each method includes adjustable parameters and real-time preview capabilities.

### Tablet Composition

The main processing pipeline arranges multiple views of a tablet into standardized layouts:

- Automatically positions obverse, reverse, side, top, and bottom views
- Preserves proper scale using color chart detection
- Arranges images in museum-standard layout
- Adds institutional logos and attribution information

### Metadata & Output

- Embeds comprehensive metadata in both TIFF and JPEG outputs
- Records photographer, institution, copyright information
- Configurable JPEG quality settings
- Customizable TIFF compression

## Background Removal Examples

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

This tool was developed for archaeological and philological research purposes.

This project is licensed under the MIT License - see the LICENSE file for details.

## Contributing

Contributions are welcome! Please feel free to submit a Pull Request.

1. Fork the repository
2. Create your feature branch (`git checkout -b feature/amazing-feature`)
3. Commit your changes (`git commit -m 'Add some amazing feature'`)
4. Push to the branch (`git push origin feature/amazing-feature`)
5. Open a Pull Request
