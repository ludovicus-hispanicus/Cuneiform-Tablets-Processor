#!/usr/bin/env python3
"""
Image processor module for the Cuneiform Tablet Processor.
Handles the core image processing functions.
"""

import os
import sys
import time
import re
import traceback
from datetime import datetime
from PyQt5.QtCore import QThread, pyqtSignal, QObject

# Image processing libraries
import cv2
import numpy as np
from PIL import Image, ImageOps, ImageDraw, ImageFont
try:
    import piexif
    from piexif import ImageIFD, ExifIFD, TAGS
    PIEXIF_AVAILABLE = True
except ImportError:
    PIEXIF_AVAILABLE = False

class ImageProcessor(QThread):
    """Worker thread to handle image processing without freezing UI"""
    progress_update = pyqtSignal(int)
    processing_complete = pyqtSignal(bool)
    error_occurred = pyqtSignal(str)
    log_message = pyqtSignal(str)
    
    def __init__(self, settings, parent=None):
        super().__init__(parent)
        self.settings = settings
        self.source_files = []
        self.tablet_groups = {}
        self.processed_count = 0
        self.total_count = 0
        self.success = True
    
    def run(self):
        """Main thread execution method"""
        try:
            self.log_message.emit("Starting image processing...")
            self.progress_update.emit(0)
            
            # Validate dependencies
            self.check_dependencies()
            
            # Organize files into tablet groups
            self.log_message.emit("Organizing input files...")
            self.organize_input_files()
            
            if not self.tablet_groups:
                self.error_occurred.emit("No valid tablet images found in the source directory.")
                self.processing_complete.emit(False)
                return
                
            # Create output directories
            tiff_dir = os.path.join(self.settings['output_path'], 'TIFF')
            if not os.path.exists(tiff_dir):
                os.makedirs(tiff_dir)
                self.log_message.emit(f"Created TIFF directory: {tiff_dir}")
                
            if self.settings['save_jpeg']:
                jpeg_dir = os.path.join(self.settings['output_path'], 'JPEG')
                if not os.path.exists(jpeg_dir):
                    os.makedirs(jpeg_dir)
                    self.log_message.emit(f"Created JPEG directory: {jpeg_dir}")
            
            # Process each tablet group
            self.total_count = len(self.tablet_groups)
            self.processed_count = 0
            
            self.log_message.emit(f"Found {self.total_count} tablet(s) to process")
            
            for tablet_number, files in self.tablet_groups.items():
                self.log_message.emit(f"Processing tablet {tablet_number}...")
                self.log_message.emit(f"Found {len(files)} image(s) for tablet {tablet_number}")
                
                try:
                    # Process this tablet
                    self.process_tablet(tablet_number, files)
                    self.processed_count += 1
                    progress = int((self.processed_count / self.total_count) * 100)
                    self.progress_update.emit(progress)
                    
                except Exception as e:
                    error_msg = f"Error processing tablet {tablet_number}: {str(e)}"
                    self.log_message.emit(error_msg)
                    # Print detailed traceback to help with debugging
                    import traceback
                    tb = traceback.format_exc()
                    self.log_message.emit(f"Traceback: {tb}")
                    self.success = False
            
            if self.success:
                self.log_message.emit("Processing complete successfully.")
            else:
                self.log_message.emit("Processing completed with errors.")
                
            self.progress_update.emit(100)
            self.processing_complete.emit(self.success)
            
        except Exception as e:
            error_msg = f"Error during processing: {str(e)}"
            self.error_occurred.emit(error_msg)
            
            # Print detailed traceback to help with debugging
            import traceback
            tb = traceback.format_exc()
            self.log_message.emit(f"Traceback: {tb}")
            
            self.processing_complete.emit(False)
    
    def check_dependencies(self):
        """Check if required libraries are available"""
        try:
            # Check OpenCV
            cv_version = cv2.__version__
            self.log_message.emit(f"OpenCV version: {cv_version}")
            
            # Check PIL
            pil_version = Image.__version__
            self.log_message.emit(f"PIL version: {pil_version}")
            
            # Check piexif - doesn't have a __version__ attribute
            self.log_message.emit(f"piexif module: {'Available' if piexif else 'Not available'}")
            
            # Check numpy
            numpy_version = np.__version__
            self.log_message.emit(f"NumPy version: {numpy_version}")
            
        except Exception as e:
            self.error_occurred.emit(f"Dependency check failed: {str(e)}")
            self.log_message.emit(f"Warning: Some dependencies might be missing or have issues, but we'll try to continue anyway.")
            # Don't raise the exception, just log the warning and continue

    def organize_input_files(self):
        """Organize input files into tablet groups"""
        self.tablet_groups = {}
        source_path = self.settings['source_path']
        
        # Supported file extensions
        valid_extensions = ['.jpg', '.jpeg', '.tif', '.tiff', '.png']
        
        self.log_message.emit(f"Scanning source directory: {source_path}")
        
        try:
            # List all items in the source directory
            items = os.listdir(source_path)
            self.log_message.emit(f"Found {len(items)} items in source directory")
            
            # First, check if all the images are already in the source folder
            # If so, we'll create a tablet group from all images in the directory
            image_files = []
            tablet_name = os.path.basename(source_path)  # Use folder name as tablet number
            
            for item in items:
                item_path = os.path.join(source_path, item)
                
                if os.path.isfile(item_path):
                    _, ext = os.path.splitext(item_path)
                    if ext.lower() in valid_extensions:
                        image_files.append(item_path)
                        self.log_message.emit(f"Found image file: {item}")
            
            # If we found image files in the source directory, create a tablet group
            if image_files:
                self.log_message.emit(f"Creating tablet group '{tablet_name}' with all files in source directory")
                self.tablet_groups[tablet_name] = image_files
                self.log_message.emit(f"Added {len(image_files)} images to tablet group '{tablet_name}'")
                
            # Otherwise, look for subdirectories with images
            else:
                for item in items:
                    item_path = os.path.join(source_path, item)
                    
                    if os.path.isdir(item_path):
                        # This is a folder, check if it contains tablet images
                        tablet_number = item  # Use folder name as tablet number
                        tablet_files = []
                        
                        self.log_message.emit(f"Checking folder: {item}")
                        
                        try:
                            folder_items = os.listdir(item_path)
                            self.log_message.emit(f"Found {len(folder_items)} items in folder {item}")
                            
                            for file in folder_items:
                                file_path = os.path.join(item_path, file)
                                _, ext = os.path.splitext(file_path)
                                
                                if os.path.isfile(file_path) and ext.lower() in valid_extensions:
                                    tablet_files.append(file_path)
                                    self.log_message.emit(f"Added file to tablet {tablet_number}: {file}")
                        
                        except Exception as e:
                            self.log_message.emit(f"Error reading folder {item_path}: {str(e)}")
                        
                        if tablet_files:
                            self.tablet_groups[tablet_number] = tablet_files
                            self.log_message.emit(f"Created tablet group for {tablet_number} with {len(tablet_files)} files")
            
            # Summarize the found tablets
            self.log_message.emit(f"Found {len(self.tablet_groups)} tablet groups to process.")
            for tablet_number, files in self.tablet_groups.items():
                self.log_message.emit(f"Tablet {tablet_number}: {len(files)} files")
                
        except Exception as e:
            self.error_occurred.emit(f"Error organizing input files: {str(e)}")
            import traceback
            tb = traceback.format_exc()
            self.log_message.emit(f"Traceback: {tb}")
            raise
    
    def process_tablet(self, tablet_number, files):
        """Process a tablet with its multiple views"""
        # Enable debug mode for this run
        self.settings['debug_mode'] = True
        
        # Create debug directory
        debug_dir = os.path.join(self.settings['output_path'], 'debug')
        if not os.path.exists(debug_dir):
            os.makedirs(debug_dir)
        
        # 1. Load all images
        images = self.load_images(files)
        if not images:
            raise ValueError(f"No valid images found for tablet {tablet_number}")
        
        # Save original view 3 (the bottom view with scale)
        for img_data in images:
            if img_data['view'] == 3:  # Bottom view with scale
                img_data['image'].save(os.path.join(debug_dir, f"{tablet_number}_original_view3.png"))
                self.log_message.emit(f"Saved original view 3 image for debugging")
        
        # 2. Composite the images into tablet views
        composite = self.compose_tablet_views(tablet_number, images, debug_dir)
        
        # Save composite before resizing
        composite.save(os.path.join(debug_dir, f"{tablet_number}_before_resize.png"))
        self.log_message.emit(f"Saved composite before resizing for debugging")
        
        # 3. Detect color scale and resize
        scaled_image = self.detect_color_scale_and_resize(composite)
        
        # Save image after resizing
        scaled_image.save(os.path.join(debug_dir, f"{tablet_number}_after_resize.png"))
        self.log_message.emit(f"Saved image after resizing for debugging")
        
        # 4. Add logo if required
        if self.settings['add_logo'] and self.settings['logo_path']:
            final_image = self.add_logo(scaled_image)
        else:
            final_image = scaled_image
        
        # Save final image before output formatting
        final_image.save(os.path.join(debug_dir, f"{tablet_number}_final_before_save.png"))
        self.log_message.emit(f"Saved final image before output formatting for debugging")
        
        # 5. Save output files with metadata
        self.save_output_files(tablet_number, final_image)
    
    def load_images(self, files):
        """Load all images for a tablet"""
        images = []
        
        for file_path in files:
            try:
                # Open image with PIL
                img = Image.open(file_path)
                
                # Extract view number from filename
                filename = os.path.basename(file_path)
                
                # Try different pattern matching for view numbers
                match = re.search(r'_(\d+)\.\w+$', filename)  # Match _01.tif pattern
                
                if match:
                    view_number = int(match.group(1))
                    self.log_message.emit(f"Detected view number {view_number} from file {filename}")
                else:
                    # If no view number in filename, try to use position in the file list
                    view_number = files.index(file_path) + 1
                    self.log_message.emit(f"No view number found in {filename}, using position: {view_number}")
                
                images.append({
                    'image': img,
                    'path': file_path,
                    'view': view_number,
                    'filename': filename
                })
                
            except Exception as e:
                self.log_message.emit(f"Error loading image {file_path}: {str(e)}")
                tb = traceback.format_exc()
                self.log_message.emit(f"Traceback: {tb}")
        
        # Check if we have enough images
        if len(images) == 0:
            self.log_message.emit("No valid images found to process")
            return []
            
        # Sort images by view number
        images.sort(key=lambda x: x['view'])
        
        self.log_message.emit(f"Loaded {len(images)} images with view numbers: {[img['view'] for img in images]}")
        return images
    
    def compose_tablet_views(self, tablet_number, images, debug_dir=None):
        """
        Compose tablet views into a single image with correct positioning
        
        View numbers correspond to:
        1: Obverse (position 5 in layer stack)
        2: Reverse (position 4)
        3: Bottom (position 3)
        4: Upper/Top (position 2)
        5: Right (position 1)
        6: Left (position 0)
        """
        # Create a larger blank canvas (8000x12000 pixels) - INCREASED HEIGHT
        canvas_width, canvas_height = 8000, 12000
        canvas = Image.new('RGB', (canvas_width, canvas_height), color='black')
        
        # Dictionary to track view positions and images
        view_positions = {}
        
        # Map view numbers to position names and their layer order
        view_map = {
            1: {'name': 'obverse', 'layer_pos': 5},
            2: {'name': 'reverse', 'layer_pos': 4},
            3: {'name': 'bottom', 'layer_pos': 3},
            4: {'name': 'upper', 'layer_pos': 2},
            5: {'name': 'right', 'layer_pos': 1},
            6: {'name': 'left', 'layer_pos': 0}
        }
        
        # Create a dictionary mapping view names to images
        view_images = {}
        
        # Log the view numbers found in the images
        found_views = []
        for img_data in images:
            view_num = img_data['view']
            found_views.append(view_num)
            if view_num in view_map:
                view_name = view_map[view_num]['name']
                view_images[view_name] = img_data['image']
                self.log_message.emit(f"Assigning view {view_num} as {view_name}")
            else:
                self.log_message.emit(f"Warning: View number {view_num} is not mapped to any position")
        
        self.log_message.emit(f"Found views: {found_views}")
        
        # If any views are missing, create small placeholder images
        for view_num, view_info in view_map.items():
            view_name = view_info['name']
            if view_name not in view_images:
                self.log_message.emit(f"Creating placeholder for missing view: {view_name} (view {view_num})")
                placeholder = Image.new('RGB', (10, 10), color='black')
                view_images[view_name] = placeholder
        
        # Save view 3 (bottom) before processing for debugging
        if debug_dir and os.path.exists(debug_dir):
            bottom_img = view_images['bottom']
            bottom_img.save(os.path.join(debug_dir, f"{tablet_number}_bottom_before_processing.png"))
            self.log_message.emit(f"Saved bottom view before processing for debugging")
        
        # ROW 1: Position Left (View 6), Obverse (View 1), Right (View 5) in the top row
        
        # 1. Position obverse (view 1) in the center top of the canvas
        obv_img = view_images['obverse']
        obv_width, obv_height = obv_img.size
        
        # Center the obverse horizontally
        obv_x = (canvas_width - obv_width) // 2
        obv_y = 0  # Top of canvas
        
        canvas.paste(obv_img, (obv_x, obv_y))
        view_positions['obverse'] = (obv_x, obv_y, obv_width, obv_height)
        self.log_message.emit(f"Positioned obverse at ({obv_x}, {obv_y}) with size {obv_width}x{obv_height}")
        
        # 2. Position left side (view 6) to the left of obverse
        left_img = view_images['left']
        left_width, left_height = left_img.size
        
        # Resize left to match obverse height
        if left_width > 10:  # Not a placeholder
            scale_factor = obv_height / left_height
            new_left_width = int(left_width * scale_factor)
            new_left_height = obv_height
            left_img = left_img.resize((new_left_width, new_left_height), Image.LANCZOS)
            self.log_message.emit(f"Resized left side to {new_left_width}x{new_left_height}")
                
        left_x = obv_x - left_img.width
        left_y = obv_y
        
        canvas.paste(left_img, (left_x, left_y))
        view_positions['left'] = (left_x, left_y, left_img.width, left_img.height)
        self.log_message.emit(f"Positioned left side at ({left_x}, {left_y})")
        
        # 3. Position right side (view 5) to the right of obverse
        right_img = view_images['right']
        right_width, right_height = right_img.size
        
        # Resize right to match obverse height
        if right_width > 10:  # Not a placeholder
            scale_factor = obv_height / right_height
            new_right_width = int(right_width * scale_factor)
            new_right_height = obv_height
            right_img = right_img.resize((new_right_width, new_right_height), Image.LANCZOS)
            self.log_message.emit(f"Resized right side to {new_right_width}x{new_right_height}")
        
        right_x = obv_x + obv_width
        right_y = obv_y
        
        canvas.paste(right_img, (right_x, right_y))
        view_positions['right'] = (right_x, right_y, right_img.width, right_img.height)
        self.log_message.emit(f"Positioned right side at ({right_x}, {right_y})")
        
        # ROW 2: Position Upper/Top (View 4) below the top row
        
        # 4. Position upper (view 4) below obverse
        upper_img = view_images['upper']
        upper_width, upper_height = upper_img.size
        
        # Resize upper to match obverse width
        if upper_width > 10:  # Not a placeholder
            scale_factor = obv_width / upper_width
            new_upper_width = obv_width
            new_upper_height = int(upper_height * scale_factor)
            upper_img = upper_img.resize((new_upper_width, new_upper_height), Image.LANCZOS)
            self.log_message.emit(f"Resized upper to {new_upper_width}x{new_upper_height}")
        
        upper_x = obv_x
        upper_y = obv_y + obv_height
        
        canvas.paste(upper_img, (upper_x, upper_y))
        view_positions['upper'] = (upper_x, upper_y, upper_img.width, upper_img.height)
        self.log_message.emit(f"Positioned upper at ({upper_x}, {upper_y})")
        
        # ROW 3: Position rotated Left, Reverse (View 2), rotated Right
        
        # 5. Create rotated copy of left side and position in row 3
        if left_img.width > 10:  # Not a placeholder
            # Duplicate and rotate left side by 180 degrees
            self.log_message.emit("Creating rotated copy of left side")
            rotated_left = left_img.copy().transpose(Image.ROTATE_180)
            
            # Position in row 3, aligned with original left
            rotated_left_x = left_x
            rotated_left_y = upper_y + upper_img.height
            
            canvas.paste(rotated_left, (rotated_left_x, rotated_left_y))
            view_positions['rotated_left'] = (rotated_left_x, rotated_left_y, rotated_left.width, rotated_left.height)
            self.log_message.emit(f"Positioned rotated left copy at ({rotated_left_x}, {rotated_left_y})")
        else:
            self.log_message.emit("Left side is a placeholder, not creating rotated copy")
            rotated_left_height = 0
        
        # 6. Position reverse (view 2) in row 3 center
        rev_img = view_images['reverse']
        rev_width, rev_height = rev_img.size
        
        # Resize reverse to match obverse width
        if rev_width > 10:  # Not a placeholder
            scale_factor = obv_width / rev_width
            new_rev_width = obv_width
            new_rev_height = int(rev_height * scale_factor)
            rev_img = rev_img.resize((new_rev_width, new_rev_height), Image.LANCZOS)
            self.log_message.emit(f"Resized reverse to {new_rev_width}x{new_rev_height}")
        
        rev_x = obv_x
        rev_y = upper_y + upper_img.height
        
        canvas.paste(rev_img, (rev_x, rev_y))
        view_positions['reverse'] = (rev_x, rev_y, rev_img.width, rev_img.height)
        self.log_message.emit(f"Positioned reverse at ({rev_x}, {rev_y})")
        
        # 7. Create rotated copy of right side and position in row 3
        if right_img.width > 10:  # Not a placeholder
            # Duplicate and rotate right side by 180 degrees
            self.log_message.emit("Creating rotated copy of right side")
            rotated_right = right_img.copy().transpose(Image.ROTATE_180)
            
            # Position in row 3, aligned with original right
            rotated_right_x = right_x
            rotated_right_y = upper_y + upper_img.height
            
            canvas.paste(rotated_right, (rotated_right_x, rotated_right_y))
            view_positions['rotated_right'] = (rotated_right_x, rotated_right_y, rotated_right.width, rotated_right.height)
            self.log_message.emit(f"Positioned rotated right copy at ({rotated_right_x}, {rotated_right_y})")
        else:
            self.log_message.emit("Right side is a placeholder, not creating rotated copy")
        
        # ROW 4: Position Bottom (View 3) below row 3
        
        # 8. Position bottom (view 3) below the reverse
        bottom_img = view_images['bottom']
        
        # Save the unmodified bottom image for debugging
        if debug_dir and os.path.exists(debug_dir):
            bottom_img.save(os.path.join(debug_dir, f"{tablet_number}_bottom_before_resize.png"))
            self.log_message.emit(f"Saved bottom view before resize for debugging")
        
        bottom_width, bottom_height = bottom_img.size
        
        # IMPORTANT CHANGE: Analyze the bottom image for color scale before resizing
        bottom_np = np.array(bottom_img)
        # Look for non-black pixels with a very low threshold
        bottom_content = np.where(bottom_np.max(axis=2) > 3)
        
        if len(bottom_content[0]) > 0 and len(bottom_content[1]) > 0:
            # Find boundaries of content in bottom image
            y_min, y_max = bottom_content[0].min(), bottom_content[0].max()
            x_min, x_max = bottom_content[1].min(), bottom_content[1].max()
            
            # Calculate height ratio of content to full image
            content_height_ratio = (y_max - y_min) / bottom_height
            self.log_message.emit(f"Bottom view content height ratio: {content_height_ratio:.4f}")
            
            # Visualize content area in bottom image
            if debug_dir and os.path.exists(debug_dir):
                debug_bottom = bottom_img.copy()
                draw = ImageDraw.Draw(debug_bottom)
                draw.rectangle([x_min, y_min, x_max, y_max], outline=(255, 0, 0), width=3)
                debug_bottom.save(os.path.join(debug_dir, f"{tablet_number}_bottom_content_area.png"))
                self.log_message.emit(f"Saved bottom view content area visualization")
        
        # MODIFIED RESIZE APPROACH:
        # Instead of resizing to match obverse width, we'll use a custom resize to keep the height
        if bottom_width > 10:  # Not a placeholder
            # Calculate width scale factor based on obverse width
            width_scale_factor = obv_width / bottom_width
            
            # Create new dimensions, scaling width while keeping relative height
            new_bottom_width = obv_width
            new_bottom_height = int(bottom_height * width_scale_factor)
            
            # If new height seems too small, it might be cutting off the scale
            if new_bottom_height < bottom_height * 0.9:
                # Add extra height to make sure we don't lose the scale
                new_bottom_height = int(new_bottom_height * 1.2)
                self.log_message.emit(f"Adding extra height to bottom view to preserve scale")
            
            self.log_message.emit(f"Resizing bottom from {bottom_width}x{bottom_height} to {new_bottom_width}x{new_bottom_height}")
            
            # Resize with LANCZOS (high quality) resampling to preserve details
            bottom_img = bottom_img.resize((new_bottom_width, new_bottom_height), Image.LANCZOS)
            
            # Save the resized bottom image for debugging
            if debug_dir and os.path.exists(debug_dir):
                bottom_img.save(os.path.join(debug_dir, f"{tablet_number}_bottom_after_resize.png"))
                self.log_message.emit(f"Saved bottom view after resize for debugging")
        
        bottom_x = obv_x
        bottom_y = rev_y + rev_img.height
        
        # Save state of canvas before pasting bottom
        if debug_dir and os.path.exists(debug_dir):
            canvas.save(os.path.join(debug_dir, f"{tablet_number}_canvas_before_bottom.png"))
            self.log_message.emit(f"Saved canvas before adding bottom view for debugging")
        
        canvas.paste(bottom_img, (bottom_x, bottom_y))
        view_positions['bottom'] = (bottom_x, bottom_y, bottom_img.width, bottom_img.height)
        self.log_message.emit(f"Positioned bottom at ({bottom_x}, {bottom_y})")
        
        # Save state of canvas after pasting bottom
        if debug_dir and os.path.exists(debug_dir):
            canvas.save(os.path.join(debug_dir, f"{tablet_number}_canvas_with_bottom.png"))
            self.log_message.emit(f"Saved canvas with bottom view for debugging")
        
        # After positioning all views, add extra space at the bottom (like in Photoshop script)
        self.log_message.emit("Adding extra space at bottom to ensure scale is preserved...")
        
        # Create a new larger canvas with extra space at the bottom (even more space than before)
        expanded_height = canvas_height + 7000  # Add 7000px at the bottom (increased from 5000px)
        expanded_canvas = Image.new('RGB', (canvas_width, expanded_height), color='black')
        
        # Paste the original canvas at the top of the expanded canvas
        expanded_canvas.paste(canvas, (0, 0))
        
        # Save expanded canvas for debugging
        if debug_dir and os.path.exists(debug_dir):
            expanded_canvas.save(os.path.join(debug_dir, f"{tablet_number}_expanded_canvas.png"))
            self.log_message.emit(f"Saved expanded canvas for debugging")
        
        # Now trim the excess canvas with more precise margins
        self.log_message.emit("Trimming excess canvas...")
        trimmed_canvas = self.trim_canvas(expanded_canvas, debug_dir, tablet_number)

        
        # Add a border of 100px
        self.log_message.emit("Adding 100px border...")
        final_canvas = ImageOps.expand(trimmed_canvas, border=100, fill='black')
        
        # Save final canvas for debugging
        if debug_dir and os.path.exists(debug_dir):
            final_canvas.save(os.path.join(debug_dir, f"{tablet_number}_final_canvas.png"))
            self.log_message.emit(f"Saved final canvas for debugging")
            
        self.log_message.emit(f"Final composite created with size {final_canvas.width}x{final_canvas.height}")
        return final_canvas

    def trim_canvas(self, image, debug_dir=None, tablet_number=None):
        """Trim excess black space from the canvas with modest margins"""
        # Convert to numpy array for processing
        np_image = np.array(image)
        
        # Get non-black pixels with slightly lower threshold to catch dark elements
        non_black = np.where(np_image.max(axis=2) > 5)  # Threshold of 5
        
        if len(non_black[0]) > 0 and len(non_black[1]) > 0:
            # Find boundaries
            y_min, y_max = non_black[0].min(), non_black[0].max()
            x_min, x_max = non_black[1].min(), non_black[1].max()
            
            # Save visualization of detected content area if debugging
            if debug_dir and os.path.exists(debug_dir) and tablet_number:
                debug_image = Image.fromarray(np_image.copy())
                draw = ImageDraw.Draw(debug_image)
                draw.rectangle([x_min, y_min, x_max, y_max], outline=(255, 0, 0), width=5)
                debug_image.save(os.path.join(debug_dir, f"{tablet_number}_detected_content_simple.png"))
                self.log_message.emit(f"Saved detected content area visualization for debugging")
            
            # Add moderate margin
            x_margin = 30  # Moderate side margin
            y_margin_top = 30  # Moderate top margin
            y_margin_bottom = 50  # Slightly larger bottom margin to ensure scale is preserved
            
            x_min = max(0, x_min - x_margin)
            y_min = max(0, y_min - y_margin_top)
            x_max = min(np_image.shape[1], x_max + x_margin)
            y_max = min(np_image.shape[0], y_max + y_margin_bottom)
            
            # Create a visualization of the trim bounds for debugging
            if debug_dir and os.path.exists(debug_dir) and tablet_number:
                debug_image = Image.fromarray(np_image.copy())
                draw = ImageDraw.Draw(debug_image)
                draw.rectangle([x_min, y_min, x_max, y_max], outline=(0, 255, 0), width=5)
                debug_image.save(os.path.join(debug_dir, f"{tablet_number}_trim_bounds_simple.png"))
                self.log_message.emit(f"Saved trim bounds visualization for debugging")
            
            # Log the trim dimensions
            self.log_message.emit(f"Trimming canvas to: ({x_min}, {y_min}), ({x_max}, {y_max})")
            self.log_message.emit(f"Original size: {image.width}x{image.height}, Content bounds: {x_max-x_min}x{y_max-y_min}")
            
            # Crop image
            cropped = image.crop((x_min, y_min, x_max, y_max))
            self.log_message.emit(f"Trimmed dimensions: {cropped.width}x{cropped.height}")
            
            # Save the cropped image for debugging
            if debug_dir and os.path.exists(debug_dir) and tablet_number:
                cropped.save(os.path.join(debug_dir, f"{tablet_number}_after_trim_simple.png"))
                self.log_message.emit(f"Saved image after trimming for debugging")
                
            return cropped
        
        return image

    def detect_color_scale_and_resize(self, image):
        """Detect color scale in image and resize appropriately with improved detection and scaling limits"""
        # Convert to numpy array for processing
        np_image = np.array(image)
        original_height, original_width = np_image.shape[:2]
        
        # Try multiple color spaces and ranges to find the blue square
        blue_square = None
        
        # First, try with LAB color space (as before)
        if blue_square is None:
            self.log_message.emit("Trying LAB color space for scale detection...")
            lab_image = cv2.cvtColor(np_image, cv2.COLOR_RGB2LAB)
            
            # Define blue color range in LAB space (wider range)
            lower_blue = np.array([0, 100, 140])
            upper_blue = np.array([70, 160, 255])
            
            blue_square = self.find_blue_square(lab_image, lower_blue, upper_blue)
        
        # If that doesn't work, try with HSV color space
        if blue_square is None:
            self.log_message.emit("Trying HSV color space for scale detection...")
            hsv_image = cv2.cvtColor(np_image, cv2.COLOR_RGB2HSV)
            
            # Blue in HSV
            lower_blue_hsv = np.array([100, 50, 50])
            upper_blue_hsv = np.array([130, 255, 255])
            
            blue_square = self.find_blue_square(hsv_image, lower_blue_hsv, upper_blue_hsv)
        
        # If still not found, try one more approach with RGB
        if blue_square is None:
            self.log_message.emit("Trying direct RGB filtering for scale detection...")
            # Look for blue in RGB space
            blue_mask = np.zeros_like(np_image[:,:,0])
            blue_mask[(np_image[:,:,2] > 150) & (np_image[:,:,0] < 100) & (np_image[:,:,1] < 100)] = 255
            
            # Find contours in the mask
            contours, _ = cv2.findContours(blue_mask.astype(np.uint8), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
            
            if contours:
                for contour in contours:
                    area = cv2.contourArea(contour)
                    if area < 100:
                        continue
                    
                    x, y, w, h = cv2.boundingRect(contour)
                    aspect_ratio = float(w) / h
                    if 0.8 <= aspect_ratio <= 1.2:
                        blue_square = (x, y, w, h)
                        self.log_message.emit(f"Found blue square using RGB filtering: {blue_square}")
                        break
        
        # Store the scale location for visualization and debugging
        if blue_square:
            x, y, w, h = blue_square
            
            # IMPORTANT: Validate the blue square size
            # If it's too small, it's probably not the actual scale but a small blue spot
            min_valid_size = 30  # Minimum side length in pixels for a valid scale
            
            if w < min_valid_size or h < min_valid_size:
                self.log_message.emit(f"Warning: Detected blue square is too small ({w}x{h}), likely not the scale.")
                self.log_message.emit("Image will not be resized to prevent extreme scaling.")
                return image
            
            self.color_scale_bbox = (x, y, w, h)
            self.log_message.emit(f"Color scale detected at: x={x}, y={y}, w={w}, h={h} (aspect ratio: {w/h:.2f})")
            
            # Create a debug image to show the detected square
            if self.settings.get('debug_mode', False):
                debug_img = Image.fromarray(np_image.copy())
                draw = ImageDraw.Draw(debug_img)
                draw.rectangle([x, y, x+w, y+h], outline=(255, 0, 0), width=3)
                debug_img.save("/tmp/scale_debug.png")
            
            # Determine if it's a 5cm or 2cm scale
            aspect_ratio = float(w) / h
            
            if aspect_ratio > 1:  # 5cm scale (color scale)
                target_width_cm = 0.77  # Target size of 0.77cm for blue square
                self.log_message.emit("Detected 5cm color scale")
            else:  # 2cm scale (gray scale)
                target_width_cm = 0.3  # Target size of 0.3cm for blue square
                self.log_message.emit("Detected 2cm gray scale")
            
            # Calculate scaling factor
            dpi = self.settings['dpi']
            pixels_per_cm = dpi / 2.54
            target_width_px = target_width_cm * pixels_per_cm
            
            scale_factor = target_width_px / w
            
            # IMPORTANT: Limit scaling factor to avoid excessive resizing
            max_scale_factor = 3.0  # Maximum allowed scaling factor
            min_scale_factor = 0.3  # Minimum allowed scaling factor
            
            if scale_factor > max_scale_factor:
                self.log_message.emit(f"Warning: Calculated scaling factor {scale_factor:.4f} is too large. Limiting to {max_scale_factor}.")
                scale_factor = max_scale_factor
            elif scale_factor < min_scale_factor:
                self.log_message.emit(f"Warning: Calculated scaling factor {scale_factor:.4f} is too small. Limiting to {min_scale_factor}.")
                scale_factor = min_scale_factor
            
            self.log_message.emit(f"Using scaling factor: {scale_factor:.4f}")
            
            # Resize the image
            new_width = int(image.width * scale_factor)
            new_height = int(image.height * scale_factor)
            
            # Sanity check on final dimensions
            max_dimension = 20000  # Maximum allowed dimension
            if new_width > max_dimension or new_height > max_dimension:
                self.log_message.emit(f"Warning: Output dimensions {new_width}x{new_height} exceed maximum allowed. Adjusting scale factor.")
                resize_ratio = min(max_dimension / new_width, max_dimension / new_height)
                new_width = int(new_width * resize_ratio)
                new_height = int(new_height * resize_ratio)
                scale_factor *= resize_ratio
                self.log_message.emit(f"Adjusted scaling factor to: {scale_factor:.4f}")
            
            self.log_message.emit(f"Resizing from {image.width}x{image.height} to {new_width}x{new_height}")
            
            try:
                resized_image = image.resize((new_width, new_height), Image.LANCZOS)
                
                # Update the color scale bbox after resizing
                self.color_scale_bbox = (
                    int(x * scale_factor),
                    int(y * scale_factor),
                    int(w * scale_factor),
                    int(h * scale_factor)
                )
                
                return resized_image
            except Exception as e:
                self.log_message.emit(f"Error during resizing: {str(e)}")
                self.log_message.emit("Returning original image without resizing")
                return image
        
        # If no blue square found, return original image
        self.log_message.emit("Warning: No color scale detected. Image will not be resized.")
        return image

    def find_blue_square(self, color_space_image, lower_range, upper_range):
        """Helper function to find blue square in a given color space"""
        # Create mask for blue region
        blue_mask = cv2.inRange(color_space_image, lower_range, upper_range)
        
        # Find contours in the mask
        contours, _ = cv2.findContours(blue_mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        
        # Debug help: save the blue mask
        if self.settings.get('debug_mode', False):
            mask_img = Image.fromarray(blue_mask)
            mask_img.save("/tmp/blue_mask.png")
        
        # Look for square-shaped contours
        for contour in contours:
            area = cv2.contourArea(contour)
            
            # Skip very small contours
            if area < 100:
                continue
            
            # Get bounding rect
            x, y, w, h = cv2.boundingRect(contour)
            
            # Check if it's approximately square
            aspect_ratio = float(w) / h
            if 0.8 <= aspect_ratio <= 1.2:
                self.log_message.emit(f"Found blue square: x={x}, y={y}, w={w}, h={h}, area={area}")
                return (x, y, w, h)
        
        return None
    
    def add_logo(self, image):
        """Add logo to the bottom of the image"""
        try:
            logo_path = self.settings['logo_path']
            if not os.path.exists(logo_path):
                self.log_message.emit(f"Warning: Logo file not found: {logo_path}")
                return image
            
            # Open logo
            logo = Image.open(logo_path)
            
            # Get dimensions
            img_width, img_height = image.size
            logo_width, logo_height = logo.size
            
            # Resize logo if it's wider than the image
            if logo_width > img_width:
                # Resize to 70% of original width
                scale_factor = 0.7 * img_width / logo_width
                new_logo_width = int(logo_width * scale_factor)
                new_logo_height = int(logo_height * scale_factor)
                logo = logo.resize((new_logo_width, new_logo_height), Image.LANCZOS)
                logo_width, logo_height = logo.size
            
            # Create a new canvas with extra space at the bottom for the logo
            new_height = img_height + logo_height + 20  # Add 20px padding
            new_image = Image.new('RGB', (img_width, new_height), color='black')
            
            # Paste original image at the top
            new_image.paste(image, (0, 0))
            
            # Center logo at the bottom
            logo_x = (img_width - logo_width) // 2
            logo_y = img_height + 10  # 10px padding
            
            # Paste logo
            new_image.paste(logo, (logo_x, logo_y), mask=logo.convert('RGBA') if logo.mode == 'RGBA' else None)
            
            return new_image
            
        except Exception as e:
            self.log_message.emit(f"Error adding logo: {str(e)}")
            return image
    
    def save_output_files(self, tablet_number, image):
        """Save the processed image as TIFF and optionally JPEG"""
        # Prepare metadata
        metadata = self.prepare_metadata(tablet_number)
        
        # Save TIFF
        tiff_dir = os.path.join(self.settings['output_path'], 'TIFF')
        tiff_path = os.path.join(tiff_dir, f"{tablet_number}.tif")
        
        # Apply compression based on settings
        compression = self.settings['compression']
        if compression == 'lzw':
            compression_type = 'tiff_lzw'
        elif compression == 'zip':
            compression_type = 'tiff_deflate'
        else:
            compression_type = None
        
        # Save TIFF with metadata
        image.save(tiff_path, format='TIFF', dpi=(self.settings['dpi'], self.settings['dpi']), 
                 compression=compression_type)
        
        # Embed IPTC metadata in TIFF
        self.embed_metadata(tiff_path, metadata)
        
        self.log_message.emit(f"Saved TIFF file: {tiff_path}")
        
        # Save JPEG if enabled
        if self.settings['save_jpeg']:
            jpeg_dir = os.path.join(self.settings['output_path'], 'JPEG')
            jpeg_path = os.path.join(jpeg_dir, f"{tablet_number}.jpg")
            
            # Save JPEG
            image.save(jpeg_path, format='JPEG', quality=self.settings['jpeg_quality'] * 8,
                      dpi=(self.settings['dpi'], self.settings['dpi']))
            
            # Embed IPTC metadata in JPEG
            self.embed_metadata(jpeg_path, metadata)
            
            self.log_message.emit(f"Saved JPEG file: {jpeg_path}")
    
    def prepare_metadata(self, tablet_number):
        """Prepare metadata dictionary for the image"""
        current_date = datetime.now().strftime("%Y:%m:%d %H:%M:%S")
        
        metadata = {
            'title': tablet_number,
            'headline': tablet_number,
            'creator': self.settings['photographer'],
            'copyright': self.settings['copyright_notice'],
            'credit': self.settings['credit_line'],
            'usage_terms': self.settings['usage_terms'],
            'source': self.settings['institution'],
            'date_created': current_date,
            'software': "Cuneiform Tablet Processor"
        }
        
        return metadata
    
    def embed_metadata(self, image_path, metadata):
        """Embed IPTC metadata into an image"""
        try:
            if not PIEXIF_AVAILABLE:
                self.log_message.emit("Warning: piexif module not available. Metadata will not be embedded.")
                return
                
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
            
        except Exception as e:
            self.log_message.emit(f"Error embedding metadata: {str(e)}")
            tb = traceback.format_exc()
            self.log_message.emit(f"Traceback: {tb}")