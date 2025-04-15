#!/usr/bin/env python3
"""
RAW file processor module for the Cuneiform Tablet Processor.
Handles conversion of RAW format images to TIFF.
"""

import os
import sys
import traceback
from datetime import datetime
from PyQt5.QtCore import QThread, pyqtSignal

class RawProcessor(QThread):
    """Worker thread to handle RAW image processing without freezing UI"""
    progress_update = pyqtSignal(int)
    processing_complete = pyqtSignal(bool)
    error_occurred = pyqtSignal(str)
    log_message = pyqtSignal(str)
    
    def __init__(self, settings, parent=None):
        super().__init__(parent)
        self.settings = settings
        self.source_path = settings.get('source_path', '')
        self.output_path = settings.get('output_path', '')
        self.raw_files = []
        self.processed_count = 0
        self.total_count = 0
        self.success = True
    
    def run(self):
        """Main thread execution method"""
        try:
            self.log_message.emit("Starting RAW processing...")
            self.progress_update.emit(0)
            
            # Check for rawpy
            self.check_rawpy()
            
            # Process RAW files
            success_count = self.process_raw_files()
            
            if success_count > 0:
                self.log_message.emit(f"RAW processing completed successfully: {success_count} files converted")
                self.processing_complete.emit(True)
            else:
                self.log_message.emit("No RAW files were processed")
                self.processing_complete.emit(False)
                
        except Exception as e:
            error_msg = f"Error during RAW processing: {str(e)}"
            self.error_occurred.emit(error_msg)
            
            # Print detailed traceback to help with debugging
            tb = traceback.format_exc()
            self.log_message.emit(f"Traceback: {tb}")
            
            self.processing_complete.emit(False)
    
    def check_rawpy(self):
        """Check if rawpy is available"""
        try:
            import rawpy
            self.log_message.emit("rawpy module is available for RAW processing")
            return True
        except ImportError:
            error_msg = "rawpy module is not installed. Please install it with 'pip install rawpy'"
            self.log_message.emit(f"Error: {error_msg}")
            self.error_occurred.emit(error_msg)
            raise ImportError(error_msg)
    
    def process_raw_files(self, source_dir=None, output_dir=None):
        """
        Process all RAW files in a directory and convert them to TIFF format
        
        Args:
            source_dir (str, optional): Directory containing RAW files. If None, uses settings source_path
            output_dir (str, optional): Directory for output TIFF files. If None, uses settings output_path
        
        Returns:
            int: Number of files successfully processed
        """
        import os
        
        # Use settings values if not specified
        if source_dir is None:
            source_dir = self.source_path
        
        if output_dir is None:
            output_dir = self.output_path
        
        if not source_dir or not os.path.exists(source_dir):
            self.log_message.emit(f"Error: Source directory does not exist: {source_dir}")
            return 0
        
        if not output_dir:
            self.log_message.emit(f"Error: Output directory not specified")
            return 0
        
        # Create output directory if it doesn't exist
        if not os.path.exists(output_dir):
            os.makedirs(output_dir)
            self.log_message.emit(f"Created output directory: {output_dir}")
        
        # Create TIFF subdirectory for output files
        tiff_dir = os.path.join(output_dir, 'TIFF')
        if not os.path.exists(tiff_dir):
            os.makedirs(tiff_dir)
            self.log_message.emit(f"Created TIFF output directory: {tiff_dir}")
        
        # Supported RAW file extensions
        raw_extensions = ['.cr2', '.nef', '.arw', '.orf', '.raf', '.raw', '.rw2', '.dng', '.3fr', '.erf']
        
        # Find all RAW files in source directory and subdirectories
        raw_files = []
        
        # First, check if RAW files are directly in the source directory
        for item in os.listdir(source_dir):
            item_path = os.path.join(source_dir, item)
            if os.path.isfile(item_path):
                _, ext = os.path.splitext(item_path)
                if ext.lower() in raw_extensions:
                    raw_files.append(item_path)
                    self.log_message.emit(f"Found RAW file: {item}")
        
        # If no RAW files in source directory, look in subdirectories
        if not raw_files:
            for root, _, files in os.walk(source_dir):
                for file in files:
                    file_path = os.path.join(root, file)
                    _, ext = os.path.splitext(file_path)
                    if ext.lower() in raw_extensions:
                        raw_files.append(file_path)
                        # Get relative path from source_dir to maintain directory structure
                        rel_path = os.path.relpath(file_path, source_dir)
                        self.log_message.emit(f"Found RAW file: {rel_path}")
        
        # Process each RAW file
        success_count = 0
        total_files = len(raw_files)
        
        self.log_message.emit(f"Found {total_files} RAW files to process")
        
        if total_files == 0:
            self.log_message.emit("No RAW files found in the specified directory")
            return 0
        
        for i, raw_file in enumerate(raw_files):
            try:
                # Determine output path, preserving directory structure
                rel_path = os.path.relpath(raw_file, source_dir)
                
                # Extract just the filename without extension
                base_name = os.path.splitext(os.path.basename(raw_file))[0]
                
                # Get parent folder name (likely tablet number)
                parent_folder = os.path.basename(os.path.dirname(raw_file))
                
                if parent_folder == os.path.basename(source_dir):
                    # If the file is directly in the source dir, just use the filename
                    output_path = os.path.join(tiff_dir, f"{base_name}.tif")
                else:
                    # If it's in a subdirectory, use the parent folder name (tablet number)
                    output_path = os.path.join(tiff_dir, f"{parent_folder}_{base_name}.tif")
                
                # Convert RAW to TIFF
                result = self.convert_raw_to_tiff(raw_file, output_path)
                
                if result:
                    success_count += 1
                
                # Update progress
                if total_files > 0:
                    progress = int(((i + 1) / total_files) * 100)
                    self.progress_update.emit(progress)
            
            except Exception as e:
                self.log_message.emit(f"Error processing RAW file {raw_file}: {str(e)}")
                tb = traceback.format_exc()
                self.log_message.emit(f"Traceback: {tb}")
        
        # Summary
        self.log_message.emit(f"Raw processing complete: {success_count} of {total_files} files converted successfully")
        return success_count
    
    def convert_raw_to_tiff(self, raw_file_path, output_file_path=None, dpi=None, compression=None):
        """
        Convert a RAW format file to TIFF format
        
        Args:
            raw_file_path (str): Path to the RAW file
            output_file_path (str, optional): Path for the output TIFF file. If None, 
                                              uses the same name as input with .tif extension
            dpi (int, optional): DPI for the output TIFF. If None, uses settings value
            compression (str, optional): Compression type ('none', 'lzw', 'zip'). If None, uses settings value
        
        Returns:
            str: Path to the created TIFF file or None on failure
        """
        try:
            import rawpy
            import numpy as np  # Ensure numpy is imported
            from PIL import Image
            import os

            self.log_message.emit(f"Converting RAW file: {raw_file_path}")

            if not os.path.exists(raw_file_path):
                self.log_message.emit(f"Error: Input RAW file does not exist: {raw_file_path}")
                return None

            if output_file_path is None:
                base_name, _ = os.path.splitext(raw_file_path)
                output_file_path = f"{base_name}.tif"

            output_dir = os.path.dirname(output_file_path)
            if output_dir and not os.path.exists(output_dir):
                os.makedirs(output_dir)

            # Use settings if not specified
            if dpi is None:
                dpi = self.settings.get('dpi', 600)
            if compression is None:
                compression = self.settings.get('compression', 'none')

            # Open and process RAW file
            self.log_message.emit("Opening RAW file with rawpy...")
            with rawpy.imread(raw_file_path) as raw:
                self.log_message.emit("Processing RAW data...")
                rgb = raw.postprocess(
                    use_camera_wb=True,  # Force camera white balance
                    output_bps=16,       # Ensure 16-bit output
                    no_auto_bright=True  # Disable auto-brightness for consistency
                )

                # Convert to uint8 if necessary (Pillow compatibility)
                if rgb.dtype == np.uint16:
                    # Scale 16-bit to 8-bit (optional: adjust scaling as needed)
                    rgb = (rgb / 256).astype(np.uint8)

                # Convert to PIL Image
                image = Image.fromarray(rgb)

                # Save as TIFF
                self.log_message.emit(f"Saving TIFF file: {output_file_path}")
                tiff_options = {
                    "dpi": (dpi, dpi),
                    "compression": compression
                }
                image.save(output_file_path, format="TIFF", **tiff_options)

                self.log_message.emit(f"Successfully converted RAW to TIFF: {output_file_path}")
                return output_file_path

        except Exception as e:
            self.log_message.emit(f"Error converting RAW to TIFF: {str(e)}")
            tb = traceback.format_exc()
            self.log_message.emit(f"Traceback: {tb}")
            return None
    
    def prepare_metadata(self, tablet_number):
        """Prepare metadata dictionary for the image"""
        # If we don't have metadata settings, return None
        if 'photographer' not in self.settings:
            return None
            
        current_date = datetime.now().strftime("%Y:%m:%d %H:%M:%S")
        
        metadata = {
            'title': tablet_number,
            'headline': tablet_number,
            'creator': self.settings.get('photographer', ''),
            'copyright': self.settings.get('copyright_notice', ''),
            'credit': self.settings.get('credit_line', ''),
            'usage_terms': self.settings.get('usage_terms', ''),
            'source': self.settings.get('institution', ''),
            'date_created': current_date,
            'software': "Cuneiform Tablet Processor"
        }
        
        return metadata
    
    def embed_metadata(self, image_path, metadata):
        """Embed IPTC metadata into an image"""
        try:
            import piexif
            from piexif import ImageIFD, ExifIFD
            
            # Load existing exif data
            try:
                exif_dict = piexif.load(image_path)
            except Exception as e:
                self.log_message.emit(f"Warning: Could not load existing EXIF data: {str(e)}")
                # Create empty EXIF dictionary
                exif_dict = {"0th": {}, "Exif": {}, "GPS": {}, "Interop": {}, "1st": {}}
            
            # Prepare zeroth IFD
            if "0th" not in exif_dict:
                exif_dict["0th"] = {}
            
            # Set basic metadata
            if metadata['title']:
                exif_dict["0th"][piexif.ImageIFD.DocumentName] = metadata['title'].encode('utf-8')
            
            if metadata['creator']:
                exif_dict["0th"][piexif.ImageIFD.Artist] = metadata['creator'].encode('utf-8')
            
            if metadata['copyright']:
                exif_dict["0th"][piexif.ImageIFD.Copyright] = metadata['copyright'].encode('utf-8')
            
            # Set software
            exif_dict["0th"][piexif.ImageIFD.Software] = metadata['software'].encode('utf-8')
            
            # Prepare Exif IFD
            if "Exif" not in exif_dict:
                exif_dict["Exif"] = {}
            
            # Set date created
            if metadata['date_created']:
                exif_dict["Exif"][piexif.ExifIFD.DateTimeOriginal] = metadata['date_created'].encode('utf-8')
            
            # Save updated exif data back to image
            try:
                exif_bytes = piexif.dump(exif_dict)
                piexif.insert(exif_bytes, image_path)
                self.log_message.emit(f"Embedded metadata in {image_path}")
            except Exception as e:
                self.log_message.emit(f"Warning: Could not embed metadata: {str(e)}")
            
        except ImportError:
            self.log_message.emit(f"Warning: piexif module not available. Metadata will not be embedded.")
        except Exception as e:
            self.log_message.emit(f"Error embedding metadata: {str(e)}")
            tb = traceback.format_exc()
            self.log_message.emit(f"Traceback: {tb}")