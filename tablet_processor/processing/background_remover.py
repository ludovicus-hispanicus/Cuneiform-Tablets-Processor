#!/usr/bin/env python3
"""
Background removal module for the Cuneiform Tablet Processor.
Removes backgrounds from tablet images using various methods.
"""

import os
import cv2
import numpy as np
from PyQt5.QtCore import QThread, pyqtSignal

try:
    from detectron2 import model_zoo
    from detectron2.engine import DefaultPredictor
    from detectron2.config import get_cfg
    import torch
    DETECTRON2_AVAILABLE = True
except ImportError:
    DETECTRON2_AVAILABLE = False

class BackgroundRemover(QThread):
    """Worker thread to handle background removal without freezing UI"""
    progress_update = pyqtSignal(int)
    processing_complete = pyqtSignal(bool)
    error_occurred = pyqtSignal(str)
    log_message = pyqtSignal(str)
        
    def __init__(self, settings, parent=None):
        super().__init__(parent)
        self.settings = settings
        self.source_path = settings.get('source_path', '')
        self.output_path = settings.get('output_path', '')
        self.method = settings.get('bg_remove_method', 2)  # Default to method 2
        self.images = []
        self.processed_count = 0
        self.total_count = 0
        self.success = True
    
    def run(self):
        """Main thread execution method"""
        try:
            self.log_message.emit("Starting background removal...")
            self.progress_update.emit(0)
            
            # Find images to process
            self.find_images()
            
            if not self.images:
                self.log_message.emit("No images found to process.")
                self.processing_complete.emit(False)
                return
                
            # Create output directory if it doesn't exist
            output_dir = os.path.join(self.output_path, 'BG_Removed')
            if not os.path.exists(output_dir):
                os.makedirs(output_dir)
                self.log_message.emit(f"Created output directory: {output_dir}")
            
            # Sort images to process _03 images last
            self.sort_images_for_processing()
            
            # Process each image
            self.total_count = len(self.images)
            self.processed_count = 0
            
            self.log_message.emit(f"Found {self.total_count} image(s) to process")
            self.log_message.emit(f"Processing non-scale images first, then scale images (_03)")
            
            for i, image_path in enumerate(self.images):
                try:
                    is_scale_image = "_03" in os.path.basename(image_path)
                    image_type = "scale" if is_scale_image else "normal"
                    self.log_message.emit(f"Processing {image_type} image {i+1}/{self.total_count}: {os.path.basename(image_path)}")
                    
                    # Determine output path
                    filename, ext = os.path.splitext(os.path.basename(image_path))
                    output_path = os.path.join(output_dir, f"{filename}_nobg{ext}")
                    
                    # Process the image
                    self.remove_background(image_path, output_path)
                    
                    self.processed_count += 1
                    progress = int((self.processed_count / self.total_count) * 100)
                    self.progress_update.emit(progress)
                    
                except Exception as e:
                    error_msg = f"Error processing image {image_path}: {str(e)}"
                    self.log_message.emit(error_msg)
                    self.success = False
            
            if self.success:
                self.log_message.emit("Background removal completed successfully.")
            else:
                self.log_message.emit("Background removal completed with errors.")
                
            self.progress_update.emit(100)
            self.processing_complete.emit(self.success)
            
        except Exception as e:
            error_msg = f"Error during background removal: {str(e)}"
            self.error_occurred.emit(error_msg)
            self.log_message.emit(error_msg)
            self.processing_complete.emit(False)
    
    def find_images(self):
        """Find all images in the source directory"""
        self.images = []
        
        if not self.source_path or not os.path.exists(self.source_path):
            self.log_message.emit(f"Source path is invalid: {self.source_path}")
            return
        
        # Supported image extensions
        valid_extensions = ['.jpg', '.jpeg', '.tif', '.tiff', '.png']
        
        # Check if it's a file or directory
        if os.path.isfile(self.source_path):
            _, ext = os.path.splitext(self.source_path)
            if ext.lower() in valid_extensions:
                self.images.append(self.source_path)
                self.log_message.emit(f"Added single file for processing: {self.source_path}")
            else:
                self.log_message.emit(f"File is not a supported image type: {self.source_path}")
        else:
            # Process all files in directory
            for root, _, files in os.walk(self.source_path):
                for file in files:
                    file_path = os.path.join(root, file)
                    _, ext = os.path.splitext(file_path)
                    if ext.lower() in valid_extensions:
                        self.images.append(file_path)
            
            self.log_message.emit(f"Found {len(self.images)} images in directory: {self.source_path}")
    
    def sort_images_for_processing(self):
        """Sort images to process _03 images last"""
        scale_images = []
        normal_images = []
        
        for image_path in self.images:
            filename = os.path.basename(image_path)
            if "_03" in filename:
                scale_images.append(image_path)
            else:
                normal_images.append(image_path)
        
        self.images = normal_images + scale_images
        self.log_message.emit(f"Processing order: {len(normal_images)} normal images, then {len(scale_images)} scale images")
    
    def remove_background(self, input_path, output_path):
        """
        Remove background from an image using method 3
        
        Args:
            input_path (str): Path to input image
            output_path (str): Path to save output image
        """
        try:
            image = cv2.imread(input_path)
            if image is None:
                self.log_message.emit(f"Error: Could not read image: {input_path}")
                return False
            
            params = self.settings.get('params', {})
            s_thresh = params.get('s_threshold', 30)
            v_thresh = params.get('v_threshold', 30)
            open_size = params.get('morph_open_size', 3)
            close_size = params.get('morph_close_size', 5)
            min_area = params.get('min_contour_area', 0.1)
            max_stray_area = params.get('max_stray_area', 0.01)
            feather_type = params.get('feather_type', 'gaussian')
            feather_amount = params.get('feather_amount', 5)
            smoothing_type = params.get('smoothing_type', None)
            smoothing_amount = params.get('smoothing_amount', 0)
            
            result = self.bg_remove_method3(
                image,
                s_threshold=s_thresh,
                v_threshold=v_thresh,
                morph_open_size=open_size,
                morph_close_size=close_size,
                min_contour_area=min_area,
                max_stray_area=max_stray_area,
                feather_type=feather_type,
                feather_amount=feather_amount,
                smoothing_type=smoothing_type,
                smoothing_amount=smoothing_amount
            )
            
            cv2.imwrite(output_path, result)
            self.log_message.emit(f"Saved background-removed image: {output_path}")
            return True
        
        except Exception as e:
            self.log_message.emit(f"Error removing background: {str(e)}")
            return False
    
    def bg_remove_method3(self, image, s_threshold=30, v_threshold=30, 
                          morph_open_size=3, morph_close_size=5, 
                          min_contour_area=0.1, max_stray_area=0.01,
                          feather_type='gaussian', feather_amount=5,
                          smoothing_type=None, smoothing_amount=0):
        """
        Enhanced background remover with feathering and smoothing options
        
        Args:
            image (ndarray): Input image (BGR format)
            s_threshold (int): Saturation threshold (0-255)
            v_threshold (int): Value threshold (0-255)
            morph_open_size (int): Kernel size for opening operation
            morph_close_size (int): Kernel size for closing operation
            min_contour_area (float): Minimum contour area as fraction of image area
            max_stray_area (float): Maximum area for stray pixels (as fraction)
            feather_type (str or None): Type of feathering ('gaussian', 'bilateral', 'morph', or None)
            feather_amount (int): Strength of feathering (1-20)
            smoothing_type (str or None): Type of smoothing ('median', 'gaussian', or None)
            smoothing_amount (int): Strength of smoothing (0-15)
            
        Returns:
            ndarray: Image with background removed (white background)
        """
        try:
            self.log_message.emit("Converting image to HSV color space")
            hsv = cv2.cvtColor(image, cv2.COLOR_BGR2HSV)
            h, s, v = cv2.split(hsv)
            img_area = image.shape[0] * image.shape[1]
            
            self.log_message.emit(f"Creating masks with s_threshold={s_threshold}, v_threshold={v_threshold}")
            s_mask = (s > s_threshold).astype(np.uint8) * 255
            v_mask = (v > v_threshold).astype(np.uint8) * 255
            combined_mask = cv2.bitwise_or(s_mask, v_mask)
            
            self.log_message.emit(f"Applying morphological open with kernel size={morph_open_size}")
            kernel_open = np.ones((morph_open_size, morph_open_size), np.uint8)
            cleaned = cv2.morphologyEx(combined_mask, cv2.MORPH_OPEN, kernel_open, iterations=1)
            
            self.log_message.emit(f"Applying morphological close with kernel size={morph_close_size}")
            kernel_close = np.ones((morph_close_size, morph_close_size), np.uint8)
            cleaned = cv2.morphologyEx(cleaned, cv2.MORPH_CLOSE, kernel_close, iterations=2)
            
            smoothed_mask = cleaned
            if smoothing_type and smoothing_amount > 0:
                kernel_size = smoothing_amount * 2 + 1
                self.log_message.emit(f"Applying {smoothing_type} smoothing with kernel size={kernel_size}")
                if smoothing_type == 'median':
                    smoothed_mask = cv2.medianBlur(cleaned, kernel_size)
                elif smoothing_type == 'gaussian':
                    smoothed_mask = cv2.GaussianBlur(cleaned, (kernel_size, kernel_size), 0)
            else:
                self.log_message.emit("Skipping smoothing (none selected or amount=0)")
            
            self.log_message.emit("Detecting contours")
            contours, _ = cv2.findContours(smoothed_mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
            final_mask = np.zeros_like(smoothed_mask)
            
            if contours:
                self.log_message.emit(f"Filtering contours with max_stray_area={max_stray_area}")
                filtered_contours = [cnt for cnt in contours 
                                   if cv2.contourArea(cnt) > (max_stray_area * img_area)]
                
                if filtered_contours:
                    self.log_message.emit("Selecting largest contour")
                    largest_contour = max(filtered_contours, key=cv2.contourArea)
                    cv2.drawContours(final_mask, [largest_contour], -1, 255, thickness=cv2.FILLED)
                    
                    self.log_message.emit("Creating blurred mask for feathering")
                    blurred_mask = final_mask.astype(np.float32) / 255.0
                    
                    if feather_type and feather_amount > 0:
                        self.log_message.emit(f"Applying {feather_type} feathering with amount={feather_amount}")
                        if feather_type == 'gaussian':
                            kernel_size = feather_amount * 2 + 1
                            blurred_mask = cv2.GaussianBlur(blurred_mask, 
                                                          (kernel_size, kernel_size), 
                                                          feather_amount)
                        elif feather_type == 'bilateral':
                            temp_mask = (blurred_mask * 255).astype(np.uint8)
                            blurred_mask = cv2.bilateralFilter(temp_mask, 
                                                             d=feather_amount * 2 + 1,
                                                             sigmaColor=feather_amount * 10,
                                                             sigmaSpace=feather_amount * 5)
                            blurred_mask = blurred_mask.astype(np.float32) / 255.0
                        elif feather_type == 'morph':
                            kernel = np.ones((feather_amount, feather_amount), np.uint8)
                            dilated = cv2.dilate(blurred_mask, kernel, iterations=1)
                            eroded = cv2.erode(blurred_mask, kernel, iterations=1)
                            gradient = dilated - eroded
                            blurred_mask = blurred_mask + gradient
                    else:
                        self.log_message.emit("Skipping feathering (none selected or amount=0)")
                    
                    self.log_message.emit("Clipping and finalizing mask")
                    blurred_mask = np.clip(blurred_mask, 0, 1)
                    final_mask = (blurred_mask * 255).astype(np.uint8)
            else:
                self.log_message.emit("No contours found")
            
            self.log_message.emit("Composing final image with white background")
            white_bg = np.ones_like(image, dtype=np.uint8) * 255
            alpha = blurred_mask
            foreground = image.astype(np.float32)
            background = white_bg.astype(np.float32)
            alpha_3c = np.repeat(alpha[:, :, np.newaxis], 3, axis=2)
            result = foreground * alpha_3c + background * (1 - alpha_3c)
            result = np.clip(result, 0, 255).astype(np.uint8)
            
            self.log_message.emit("Background removal completed for image")
            return result
        
        except Exception as e:
            self.log_message.emit(f"Error in method3 with feathering/smoothing: {str(e)}")
            # Simple fallback - basic threshold on value channel
            try:
                self.log_message.emit("Attempting fallback: basic threshold on value channel")
                hsv = cv2.cvtColor(image, cv2.COLOR_BGR2HSV)
                _, _, v = cv2.split(hsv)
                _, mask = cv2.threshold(v, 30, 255, cv2.THRESH_BINARY)
                white_bg = np.ones_like(image) * 255
                foreground = cv2.bitwise_and(image, image, mask=mask)
                bg_mask = cv2.bitwise_not(mask)
                background = cv2.bitwise_and(white_bg, white_bg, mask=bg_mask)
                result = cv2.add(foreground, background)
                self.log_message.emit("Fallback completed")
                return result
            except:
                self.log_message.emit("Fallback failed, returning original image")
                return image