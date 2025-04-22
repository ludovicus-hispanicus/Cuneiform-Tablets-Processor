#!/usr/bin/env python3
"""
Background removal module for the Cuneiform Tablet Processor.
Removes backgrounds from tablet images using various methods.
"""

import os
import sys
import traceback
import cv2
import numpy as np
from PyQt5.QtCore import QThread, pyqtSignal
from PIL import Image

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
                    tb = traceback.format_exc()
                    self.log_message.emit(f"Traceback: {tb}")
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
            tb = traceback.format_exc()
            self.log_message.emit(f"Traceback: {tb}")
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
        # Split images into two groups: _03 images and other images
        scale_images = []
        normal_images = []
        
        for image_path in self.images:
            filename = os.path.basename(image_path)
            if "_03" in filename:
                scale_images.append(image_path)
            else:
                normal_images.append(image_path)
        
        # Combine the groups - normal first, then scale
        self.images = normal_images + scale_images
        
        # Log the processing order
        self.log_message.emit(f"Processing order: {len(normal_images)} normal images, then {len(scale_images)} scale images")
    
    def remove_background(self, input_path, output_path):
        """
        Remove background from an image using the selected method
        
        Args:
            input_path (str): Path to input image
            output_path (str): Path to save output image
        """
        try:
            # Read the image
            image = cv2.imread(input_path)
            if image is None:
                self.log_message.emit(f"Error: Could not read image: {input_path}")
                return False
            
            # Get parameters if provided
            params = self.settings.get('params', {})
            
            # Process based on selected method
            if self.method == 1:
                    clahe_clip = params.get('clahe_clip', 4.0)
                    block_size = params.get('block_size', 15)
                    c_constant = params.get('c_constant', 2)
                    dilate_iterations = params.get('dilate_iterations', 1)
                    
                    result = self.bg_remove_method1(
                        image,
                        clahe_clip=clahe_clip,
                        block_size=block_size,
                        c_constant=c_constant,
                        dilate_iterations=dilate_iterations
                    )
            elif self.method == 2:
                block_size = params.get('block_size', 15)
                c_constant = params.get('c_constant', 2)
                pre_blur = params.get('pre_blur', 5)
                post_blur = params.get('post_blur', 0)
                result = self.bg_remove_method2(
                    image, 
                    block_size=block_size, 
                    c_constant=c_constant,
                    pre_blur=pre_blur,
                    post_blur=post_blur
                )
            elif self.method == 3:
                s_thresh = params.get('s_threshold', 30)
                v_thresh = params.get('v_threshold', 30)
                open_size = params.get('morph_open_size', 3)
                close_size = params.get('morph_close_size', 5)
                min_area = params.get('min_contour_area', 0.1)
                debug = params.get('debug', False)
                
                result = self.bg_remove_method3(
                    image,
                    s_threshold=s_thresh,
                    v_threshold=v_thresh,
                    morph_open_size=open_size,
                    morph_close_size=close_size,
                    min_contour_area=min_area,
                    debug=debug
                )
            elif self.method == 4:
                block_size = params.get('block_size', 11)
                c_constant = params.get('c_constant', 2)
                result = self.bg_remove_method4(image, block_size, c_constant)
            elif self.method == 5:
                low_threshold = params.get('low_threshold', 30)
                high_threshold = params.get('high_threshold', 100)
                dilation_iterations = params.get('dilation_iterations', 3)
                result = self.bg_remove_method5(image, low_threshold, high_threshold, dilation_iterations)
            
            elif self.method == 8:
                model_name = params.get('model_name', 'u2net')
                alpha_matting = params.get('alpha_matting', True)
                alpha_matting_foreground_threshold = params.get('alpha_matting_foreground_threshold', 240)
                alpha_matting_background_threshold = params.get('alpha_matting_background_threshold', 10)
                alpha_matting_erode_size = params.get('alpha_matting_erode_size', 10)
                
                result = self.bg_remove_rembg(
                    image,
                    model_name=model_name,
                    alpha_matting=alpha_matting,
                    alpha_matting_foreground_threshold=alpha_matting_foreground_threshold,
                    alpha_matting_background_threshold=alpha_matting_background_threshold,
                    alpha_matting_erode_size=alpha_matting_erode_size
                )
            elif self.method == 9:
                result = self.bg_remove_otsu_contour(image)    
            elif self.method == 10:
                # Make sure this calls the DeepLabV3 method, not otsu_contour
                result = self.bg_remove_ml(image)
            elif self.method == 11:  # New Detectron2 method
                confidence = params.get('detectron_confidence', 0.7)
                self.log_message.emit(f"Using Detectron2 with confidence: {confidence}")
                result = self.bg_remove_detectron2(image, confidence)  # Make sure this matches the actual method name
            
            else:
                result = self.bg_remove_method2(image)  # Default to method 2
            
            # Save the result
            cv2.imwrite(output_path, result)
            self.log_message.emit(f"Saved background-removed image: {output_path}")
            
            return True
        
        except Exception as e:
            self.log_message.emit(f"Error removing background: {str(e)}")
            tb = traceback.format_exc()
            self.log_message.emit(f"Traceback: {tb}")
            return False
    
    def bg_remove_method1(self, image, clahe_clip=4.0, block_size=15, c_constant=2, dilate_iterations=1):
        """
        Background removal using adaptive thresholding with preprocessing
        optimized for cuneiform tablets against dark backgrounds
        
        Args:
            image (ndarray): Input image
            clahe_clip (float): CLAHE clip limit for contrast enhancement
            block_size (int): Block size for adaptive thresholding
            c_constant (int): Constant subtracted from mean in thresholding
            dilate_iterations (int): Number of dilation iterations for final mask
        
        Returns:
            ndarray: Image with background removed (white background)
        """
        # Convert to grayscale
        gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
        
        # Apply CLAHE for local contrast enhancement
        clahe = cv2.createCLAHE(clipLimit=clahe_clip, tileGridSize=(8,8))
        enhanced = clahe.apply(gray)
        
        # Apply bilateral filter to reduce noise while preserving edges
        blurred = cv2.bilateralFilter(enhanced, 7, 50, 50)
        
        # Two-stage thresholding process
        _, rough_mask = cv2.threshold(blurred, 0, 255, cv2.THRESH_BINARY_INV + cv2.THRESH_OTSU)
        
        # Ensure block_size is odd
        if block_size % 2 == 0:
            block_size += 1
            
        adaptive_thresh = cv2.adaptiveThreshold(
            blurred, 
            255, 
            cv2.ADAPTIVE_THRESH_GAUSSIAN_C, 
            cv2.THRESH_BINARY_INV, 
            block_size,
            c_constant
        )
        
        # Combine masks
        combined_mask = cv2.bitwise_and(rough_mask, adaptive_thresh)
        
        # Morphological operations
        kernel = np.ones((3,3), np.uint8)
        cleaned = cv2.morphologyEx(combined_mask, cv2.MORPH_OPEN, kernel, iterations=1)
        cleaned = cv2.morphologyEx(cleaned, cv2.MORPH_CLOSE, kernel, iterations=2)
        
        # Find contours
        contours, _ = cv2.findContours(cleaned, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        
        if contours:
            largest_contour = max(contours, key=cv2.contourArea)
            mask = np.zeros_like(gray)
            cv2.drawContours(mask, [largest_contour], -1, 255, thickness=cv2.FILLED)
            
            # Dilate based on parameter
            dilate_kernel = np.ones((3,3), np.uint8)
            mask = cv2.dilate(mask, dilate_kernel, iterations=dilate_iterations)
        else:
            mask = cleaned
        
        # Create white background
        white_bg = np.ones_like(image) * 255
        
        # Extract foreground and background
        foreground = cv2.bitwise_and(image, image, mask=mask)
        bg_mask = cv2.bitwise_not(mask)
        background = cv2.bitwise_and(white_bg, white_bg, mask=bg_mask)
        
        # Combine
        result = cv2.add(foreground, background)
        
        return result

    def bg_remove_method2(self, image, block_size=15, c_constant=2, pre_blur=5, post_blur=0):
        """
        Improved background remover method 2 for cuneiform tablets against dark backgrounds
        
        Args:
            image (ndarray): Input image
            block_size (int): Block size for adaptive threshold
            c_constant (int): Constant subtracted from mean
            pre_blur (int): Gaussian blur kernel size before thresholding (0 to disable)
            post_blur (int): Gaussian blur kernel size for final mask (0 to disable)
        
        Returns:
            ndarray: Image with background removed
        """
        # Convert to Grayscale
        image_grey = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
        
        # Apply CLAHE for contrast enhancement
        clahe = cv2.createCLAHE(clipLimit=3.0, tileGridSize=(8,8))
        image_grey = clahe.apply(image_grey)
        
        # Apply pre-processing blur if specified
        if pre_blur > 0:
            # Ensure kernel size is odd
            if pre_blur % 2 == 0:
                pre_blur += 1
            image_grey = cv2.GaussianBlur(image_grey, (pre_blur, pre_blur), 0)

        # Make sure block_size is odd
        if block_size % 2 == 0:
            block_size += 1
            
        # Use adaptive thresholding with customizable parameters
        mask = cv2.adaptiveThreshold(
            image_grey, 
            255, 
            cv2.ADAPTIVE_THRESH_GAUSSIAN_C, 
            cv2.THRESH_BINARY_INV, 
            block_size,
            c_constant
        )
        
        # Clean up the mask with morphological operations
        kernel = np.ones((3, 3), np.uint8)
        mask = cv2.morphologyEx(mask, cv2.MORPH_OPEN, kernel, iterations=1)
        mask = cv2.morphologyEx(mask, cv2.MORPH_CLOSE, kernel, iterations=2)
        
        # Find contours to get complete tablet shape
        contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        
        # If contours found, use the largest one as mask
        if contours and len(contours) > 0:
            # Get the largest contour
            largest_contour = max(contours, key=cv2.contourArea)
            
            # Create a filled contour mask
            contour_mask = np.zeros_like(image_grey)
            cv2.drawContours(contour_mask, [largest_contour], -1, 255, thickness=cv2.FILLED)
            
            # Combine with original mask to get both the outline and internal details
            final_mask = cv2.bitwise_or(mask, contour_mask)
        else:
            final_mask = mask
        
        # Apply post-blur to smooth mask edges if specified
        if post_blur > 0:
            if post_blur % 2 == 0:
                post_blur += 1
            final_mask = cv2.GaussianBlur(final_mask, (post_blur, post_blur), 0)
            _, final_mask = cv2.threshold(final_mask, 127, 255, cv2.THRESH_BINARY)
        
        # Create white background
        white_bg = np.ones_like(image) * 255
        
        # Apply mask to extract foreground
        foreground = cv2.bitwise_and(image, image, mask=final_mask)
        
        # Invert mask for background
        bg_mask = cv2.bitwise_not(final_mask)
        background = cv2.bitwise_and(white_bg, white_bg, mask=bg_mask)
        
        # Combine foreground and background
        final_image = cv2.add(foreground, background)
        
        return final_image

    def bg_remove_method3(self, image, s_threshold=30, v_threshold=30, 
                          morph_open_size=3, morph_close_size=5, 
                          min_contour_area=0.1, max_stray_area=0.01,
                          feather_type='gaussian', feather_amount=5,
                          debug=False, debug_output_folder=None):
        """
        Enhanced background remover with feathering options for smooth edges
        
        Args:
            image (ndarray): Input image (BGR format)
            s_threshold (int): Saturation threshold (0-255)
            v_threshold (int): Value threshold (0-255)
            morph_open_size (int): Kernel size for opening operation
            morph_close_size (int): Kernel size for closing operation
            min_contour_area (float): Minimum contour area as fraction of image area
            max_stray_area (float): Maximum area for stray pixels (as fraction)
            feather_type (str): Type of feathering ('gaussian', 'bilateral', 'morph')
            feather_amount (int): Strength of feathering (1-20)
            debug (bool): Whether to save intermediate images
            debug_output_folder (str): Folder to save debug images
            
        Returns:
            ndarray: Image with background removed (white background)
        """
        try:
            if debug and debug_output_folder:
                os.makedirs(debug_output_folder, exist_ok=True)
            
            # Convert to HSV and create initial mask (existing code)
            hsv = cv2.cvtColor(image, cv2.COLOR_BGR2HSV)
            h, s, v = cv2.split(hsv)
            img_area = image.shape[0] * image.shape[1]
            
            s_mask = (s > s_threshold).astype(np.uint8) * 255
            v_mask = (v > v_threshold).astype(np.uint8) * 255
            combined_mask = cv2.bitwise_or(s_mask, v_mask)
            
            # Morphological operations (existing code)
            kernel_open = np.ones((morph_open_size, morph_open_size), np.uint8)
            cleaned = cv2.morphologyEx(combined_mask, cv2.MORPH_OPEN, kernel_open, iterations=1)
            kernel_close = np.ones((morph_close_size, morph_close_size), np.uint8)
            cleaned = cv2.morphologyEx(cleaned, cv2.MORPH_CLOSE, kernel_close, iterations=2)
            
            # Find contours and create final mask (existing code)
            contours, _ = cv2.findContours(cleaned, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
            final_mask = np.zeros_like(cleaned)
            
            if contours:
                filtered_contours = [cnt for cnt in contours 
                                   if cv2.contourArea(cnt) > (max_stray_area * img_area)]
                
                if filtered_contours:
                    largest_contour = max(filtered_contours, key=cv2.contourArea)
                    cv2.drawContours(final_mask, [largest_contour], -1, 255, thickness=cv2.FILLED)
                    
                    # ===== NEW FEATHERING CODE =====
                    # Convert mask to float for smooth operations
                    final_mask_float = final_mask.astype(np.float32) / 255.0
                    
                    if feather_type == 'gaussian':
                        # Gaussian blur feathering
                        kernel_size = feather_amount * 2 + 1  # Ensure odd number
                        blurred_mask = cv2.GaussianBlur(final_mask_float, 
                                                      (kernel_size, kernel_size), 
                                                      feather_amount)
                    
                    elif feather_type == 'bilateral':
                        # Bilateral filter preserves edges better
                        blurred_mask = cv2.bilateralFilter(final_mask_float, 
                                                         d=feather_amount * 2 + 1,
                                                         sigmaColor=feather_amount * 10,
                                                         sigmaSpace=feather_amount * 5)
                    
                    elif feather_type == 'morph':
                        # Morphological gradient feathering
                        kernel = np.ones((feather_amount, feather_amount), np.uint8)
                        dilated = cv2.dilate(final_mask_float, kernel, iterations=1)
                        eroded = cv2.erode(final_mask_float, kernel, iterations=1)
                        gradient = dilated - eroded
                        blurred_mask = final_mask_float + (gradient * 0.5)
                    
                    # Ensure values stay in 0-1 range
                    blurred_mask = np.clip(blurred_mask, 0, 1)
                    
                    # Convert back to 8-bit mask
                    final_mask = (blurred_mask * 255).astype(np.uint8)
                    # ===== END FEATHERING CODE =====
                    
                    if debug and debug_output_folder:
                        cv2.imwrite(f"{debug_output_folder}/6_feathered_mask.png", final_mask)
            
            # Create white background and apply mask
            white_bg = np.ones_like(image) * 255
            foreground = cv2.bitwise_and(image, image, mask=final_mask)
            bg_mask = cv2.bitwise_not(final_mask)
            background = cv2.bitwise_and(white_bg, white_bg, mask=bg_mask)
            result = cv2.add(foreground, background)
            
            return result
            
        except Exception as e:
            self.log_message.emit(f"Error in method3 with feathering: {str(e)}")
            return self.bg_remove_method2(image)
    
    def bg_remove_method4(self, image, block_size=11, c_constant=2):
        """
        Improved method for removing black background from cuneiform tablets
        
        Args:
            image (ndarray): Input image with black background
            block_size (int): Block size for adaptive threshold
            c_constant (int): Constant subtracted from mean
        
        Returns:
            ndarray: Image with background replaced by white
        """
        # Convert to grayscale
        gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
        
        # Apply CLAHE to enhance contrast
        clahe = cv2.createCLAHE(clipLimit=3.0, tileGridSize=(8,8))
        enhanced = clahe.apply(gray)
        
        # Apply bilateral filter to reduce noise while preserving edges
        blurred = cv2.bilateralFilter(enhanced, 9, 75, 75)
        
        # Apply adaptive thresholding with adjusted parameters
        # The key is to use a small C value for dark objects on dark backgrounds
        binary = cv2.adaptiveThreshold(
            blurred, 
            255, 
            cv2.ADAPTIVE_THRESH_GAUSSIAN_C, 
            cv2.THRESH_BINARY_INV, 
            max(3, (block_size if block_size % 2 == 1 else block_size + 1)),  # Ensure odd value
            max(1, c_constant - 1)  # Use smaller C value for better detection
        )
        
        # Clean up the mask - remove noise
        kernel = np.ones((3, 3), np.uint8)
        binary = cv2.morphologyEx(binary, cv2.MORPH_OPEN, kernel, iterations=1)
        binary = cv2.morphologyEx(binary, cv2.MORPH_CLOSE, kernel, iterations=3)  # More closing to fill in gaps
        
        # Find contours to find the full tablet shape
        contours, _ = cv2.findContours(binary, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        
        # Create a new mask from the largest contour (should be the tablet)
        mask = np.zeros_like(gray)
        if contours:
            largest_contour = max(contours, key=cv2.contourArea)
            cv2.drawContours(mask, [largest_contour], 0, 255, thickness=cv2.FILLED)
            
            # Dilate the mask slightly to ensure we get the entire tablet
            kernel = np.ones((5, 5), np.uint8)
            mask = cv2.dilate(mask, kernel, iterations=1)
        else:
            # If no contour found, fall back to the binary threshold
            mask = binary
        
        # Create white background
        white_bg = np.ones_like(image) * 255
        
        # Apply mask to extract foreground
        foreground = cv2.bitwise_and(image, image, mask=mask)
        
        # Invert mask for background
        bg_mask = cv2.bitwise_not(mask)
        background = cv2.bitwise_and(white_bg, white_bg, mask=bg_mask)
        
        # Combine foreground and background
        result = cv2.add(foreground, background)
        
        return result
    
    def bg_remove_method5(self, image, low_threshold=0, high_threshold=30, dilation_iterations=1, min_contour_area=0):
        """
        Background remover method 5 using edge detection and contour finding
        
        Args:
            image (ndarray): Input image
            low_threshold (int): Lower threshold for Canny edge detection (default: 30)
            high_threshold (int): Higher threshold for Canny edge detection (default: 100)
            dilation_iterations (int): Number of iterations for dilation (default: 3)
            min_contour_area (float): Minimum contour area to keep (default: 0, keeps all)
        
        Returns:
            ndarray: Image with background removed
        """
        # Convert to grayscale
        gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
        
        # Apply edge detection with customizable thresholds
        edges = cv2.Canny(gray, low_threshold, high_threshold)
        
        # Dilate to close gaps in edges with customizable iterations
        kernel = np.ones((5, 5), np.uint8)
        dilated = cv2.dilate(edges, kernel, iterations=dilation_iterations)
        
        # Find contours
        contours, _ = cv2.findContours(dilated, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        
        # Filter contours by size if specified
        if min_contour_area > 0:
            contours = [c for c in contours if cv2.contourArea(c) > min_contour_area]
        
        # Create a mask from the largest contour (presumably the tablet)
        mask = np.zeros_like(gray)
        if contours:
            largest_contour = max(contours, key=cv2.contourArea)
            cv2.drawContours(mask, [largest_contour], 0, 255, -1)
        
        # White background
        white_bg = np.ones_like(image) * 255
        
        # Apply mask to original image to get foreground
        foreground = cv2.bitwise_and(image, image, mask=mask)
        
        # Invert mask for background
        bg_mask = cv2.bitwise_not(mask)
        background = cv2.bitwise_and(white_bg, white_bg, mask=bg_mask)
        
        # Combine foreground and background
        result = cv2.add(foreground, background)
        
        return result

           
    def bg_remove_rembg(self, image, model_name='u2net', alpha_matting=True, 
                                alpha_matting_foreground_threshold=240,
                                alpha_matting_background_threshold=10,
                                alpha_matting_erode_size=10):
        """
        Enhanced Rembg background remover with configurable parameters
        
        Args:
            image (ndarray): Input image (BGR format from OpenCV)
            model_name (str): Model to use ('u2net', 'u2netp', 'u2net_human_seg', 'silueta')
            alpha_matting (bool): Whether to use alpha matting
            alpha_matting_foreground_threshold (int): Alpha matting foreground threshold
            alpha_matting_background_threshold (int): Alpha matting background threshold
            alpha_matting_erode_size (int): Alpha matting erode size
        
        Returns:
            ndarray: Image with background removed (white background)
        """
        try:
            # Import rembg
            from rembg import remove, new_session
            import PIL.Image as PILImage
            from io import BytesIO
            
            # Convert OpenCV image (BGR) to PIL Image (RGB)
            image_rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
            pil_image = PILImage.fromarray(image_rgb)
            
            # Create session with specific model
            session = new_session(model_name)
            
            # Process with rembg using custom parameters
            # For dark/black backgrounds, lower the alpha_matting_background_threshold
            output_pil = remove(
                pil_image,
                session=session,
                alpha_matting=alpha_matting,
                alpha_matting_foreground_threshold=alpha_matting_foreground_threshold,
                alpha_matting_background_threshold=alpha_matting_background_threshold,
                alpha_matting_erode_size=alpha_matting_erode_size
            )
            
            # Extract alpha channel and create white background
            alpha = output_pil.split()[-1]
            bg = PILImage.new("RGB", output_pil.size, (255, 255, 255))
            bg.paste(output_pil, mask=alpha)
            
            # Convert back to OpenCV format
            result = cv2.cvtColor(np.array(bg), cv2.COLOR_RGB2BGR)
            
            return result
        except ImportError:
            self.log_message.emit("Rembg library not installed. Install with 'pip install rembg'")
            return self.bg_remove_method6(image, k=2, bg_detection_mode='darkest')
    
    def bg_remove_ml(self, image):
        """
        Background remover using a pre-trained DeepLabV3 model with enhanced debugging
        """
        try:
            # Import necessary libraries
            import torch
            import torch.nn.functional as F
            from torchvision import transforms
            import matplotlib.pyplot as plt
            import os
            
            # Create debug directory
            debug_dir = os.path.join(self.output_path, "debug_ml")
            os.makedirs(debug_dir, exist_ok=True)
            
            self.log_message.emit(f"Starting ML-based segmentation with debugging to {debug_dir}")
            
            # Save original image for debugging
            cv2.imwrite(os.path.join(debug_dir, "01_original.jpg"), image)
            
            # Check if we can use GPU
            device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
            self.log_message.emit(f"Using device: {device}")
            
            # Load pre-trained model
            model = torch.hub.load('pytorch/vision', 'deeplabv3_resnet101', weights=True)
            model.to(device)
            model.eval()
            
            # Prepare the image
            input_image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
            preprocess = transforms.Compose([
                transforms.ToPILImage(),
                transforms.Resize((520, 520)),
                transforms.ToTensor(),
                transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
            ])
            input_tensor = preprocess(input_image)
            
            # Save resized input for debugging
            resized_np = input_tensor.permute(1, 2, 0).numpy() * 255
            cv2.imwrite(os.path.join(debug_dir, "02_resized_input.jpg"), cv2.cvtColor(resized_np.astype(np.uint8), cv2.COLOR_RGB2BGR))
            
            input_batch = input_tensor.unsqueeze(0).to(device)
            
            # Get prediction
            self.log_message.emit("Running model inference...")
            with torch.no_grad():
                output = model(input_batch)['out'][0]
            
            self.log_message.emit("Model inference complete")
            
            # Get probabilities
            probabilities = F.softmax(output, dim=0).cpu().numpy()
            
            # Get class predictions
            predictions = output.argmax(0).byte().cpu().numpy()
            
            # Save prediction visualization for debugging
            plt.figure(figsize=(10, 10))
            plt.imshow(predictions)
            plt.colorbar()
            plt.title("Class Predictions")
            plt.savefig(os.path.join(debug_dir, "03_class_predictions.png"))
            plt.close()
            
            # Log unique classes and their counts
            unique_classes, counts = np.unique(predictions, return_counts=True)
            self.log_message.emit(f"Classes found: {list(zip(unique_classes, counts))}")
            
            # Try more potential classes, including some that might remotely resemble tablets
            potential_classes = [1, 2, 3, 7, 13, 14, 15, 18, 19, 20, 44, 45, 62, 67, 73, 77, 84]

            # Lower the minimum pixel count to accept a class
            if class_pixels > 50:  # Lower from current value
                self.log_message.emit(f"Class {class_id} has {class_pixels} pixels")
                mask = np.logical_or(mask, class_mask)
                classes_used.append(class_id)
            
            # Create a combined mask
            mask = np.zeros_like(predictions, dtype=np.uint8)
            classes_used = []
            
            for class_id in potential_classes:
                if class_id in unique_classes:
                    class_mask = predictions == class_id
                    class_pixels = np.sum(class_mask)
                    
                    if class_pixels > 0:
                        self.log_message.emit(f"Class {class_id} has {class_pixels} pixels")
                        mask = np.logical_or(mask, class_mask)
                        classes_used.append(class_id)
                        
                        # Save individual class masks for debugging
                        class_mask_vis = class_mask.astype(np.uint8) * 255
                        cv2.imwrite(os.path.join(debug_dir, f"04_class_{class_id}_mask.png"), class_mask_vis)
            
            # If no classes matched, use the largest non-background class
            if not classes_used and len(unique_classes) > 1:
                # Usually class 0 is background
                non_bg_classes = [(c, counts[i]) for i, c in enumerate(unique_classes) if c > 0]
                
                if non_bg_classes:
                    # Sort by count (descending)
                    non_bg_classes.sort(key=lambda x: x[1], reverse=True)
                    largest_class = non_bg_classes[0][0]
                    
                    self.log_message.emit(f"No tablet classes matched. Using largest non-background class: {largest_class}")
                    mask = predictions == largest_class
                    classes_used.append(largest_class)
            
            # If still no mask, fallback to traditional method
            if not classes_used:
                self.log_message.emit("No suitable classes found. Falling back to traditional method.")
                
                # Save an empty mask for debugging
                empty_mask = np.zeros_like(predictions, dtype=np.uint8)
                cv2.imwrite(os.path.join(debug_dir, "05_empty_mask.png"), empty_mask)
                
                return self.bg_remove_method2(image)
            
            # Create binary mask
            binary_mask = mask.astype(np.uint8) * 255
            
            # Save the combined mask for debugging
            cv2.imwrite(os.path.join(debug_dir, "06_combined_mask.png"), binary_mask)
            
            # Resize mask to original size
            h, w = image.shape[:2]
            resized_mask = cv2.resize(binary_mask, (w, h), interpolation=cv2.INTER_NEAREST)
            
            # Save the resized mask for debugging
            cv2.imwrite(os.path.join(debug_dir, "07_resized_mask.png"), resized_mask)
            
            # Clean up mask with morphological operations
            kernel = np.ones((5, 5), np.uint8)
            cleaned_mask = cv2.morphologyEx(resized_mask, cv2.MORPH_CLOSE, kernel, iterations=2)
            
            # Save the cleaned mask for debugging
            cv2.imwrite(os.path.join(debug_dir, "08_cleaned_mask.png"), cleaned_mask)
            
            # Create white background
            white_bg = np.ones_like(image) * 255
            
            # Apply mask to extract foreground
            foreground = cv2.bitwise_and(image, image, mask=cleaned_mask)
            
            # Save the foreground for debugging
            cv2.imwrite(os.path.join(debug_dir, "09_foreground.jpg"), foreground)
            
            # Invert mask for background
            bg_mask = cv2.bitwise_not(cleaned_mask)
            
            # Save the background mask for debugging
            cv2.imwrite(os.path.join(debug_dir, "10_bg_mask.png"), bg_mask)
            
            background = cv2.bitwise_and(white_bg, white_bg, mask=bg_mask)
            
            # Save the background for debugging
            cv2.imwrite(os.path.join(debug_dir, "11_background.jpg"), background)
            
            # Combine foreground and background
            result = cv2.add(foreground, background)
            
            # Save the final result for debugging
            cv2.imwrite(os.path.join(debug_dir, "12_final_result.jpg"), result)
            
            self.log_message.emit(f"ML processing complete using classes: {classes_used}")
            
            return result
        
        except Exception as e:
            self.log_message.emit(f"ML processing failed: {str(e)}")
            self.log_message.emit(traceback.format_exc())
            return self.bg_remove_method2(image)
        
    def bg_remove_detectron2(self, image, confidence=0.7, class_ids=None, transparent_bg=False):
        """
        Remove background using Detectron2's Mask R-CNN"""
        self.log_message.emit(f"Detectron2 received confidence parameter: {confidence}")
        """Args:
            image: Input image (BGR)
            confidence: Detection confidence threshold
            class_ids: List of COCO class IDs to consider (None = all)
            transparent_bg: Whether to create transparent background
            
        Returns:
            Image with background removed
        """
        if not DETECTRON2_AVAILABLE:
            self.log_message.emit("Detectron2 is not installed. Falling back to method 5.")
            return self.bg_remove_method5(image)
            
        try:
            # Initialize configuration
            cfg = get_cfg()
            cfg.merge_from_file(model_zoo.get_config_file(
                "COCO-InstanceSegmentation/mask_rcnn_R_50_FPN_3x.yaml"))
            cfg.MODEL.WEIGHTS = model_zoo.get_checkpoint_url(
                "COCO-InstanceSegmentation/mask_rcnn_R_50_FPN_3x.yaml")
            cfg.MODEL.ROI_HEADS.SCORE_THRESH_TEST = confidence
            cfg.MODEL.DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
            
            # Log which device we're using
            self.log_message.emit(f"Using Detectron2 on {cfg.MODEL.DEVICE}")
            
            # Process image
            predictor = DefaultPredictor(cfg)
            outputs = predictor(image)
            
            # Get instances
            instances = outputs["instances"].to("cpu")
            if len(instances) == 0:
                self.log_message.emit("No objects detected. Falling back to method 5.")
                return self.bg_remove_method5(image)
                
            # Get predictions
            pred_classes = instances.pred_classes.numpy()
            pred_scores = instances.scores.numpy()
            pred_masks = instances.pred_masks.numpy()
            
            # Filter by class if specified
            if class_ids is not None:
                class_indices = [i for i, cls in enumerate(pred_classes) if cls in class_ids]
                if not class_indices:
                    self.log_message.emit(f"No objects of specified classes found. Detected classes: {np.unique(pred_classes)}")
                    return self.bg_remove_method5(image)
                    
                # Filter to only keep specified classes
                filtered_scores = pred_scores[class_indices]
                filtered_masks = pred_masks[class_indices]
                
                # Find best scoring instance
                best_idx = np.argmax(filtered_scores)
                best_mask = filtered_masks[best_idx]
            else:
                # No class filtering - use highest scoring instance
                best_idx = np.argmax(pred_scores)
                best_mask = pred_masks[best_idx]
                best_class = pred_classes[best_idx]
                best_score = pred_scores[best_idx]
                
                # Log what was detected
                coco_classes = ["person", "bicycle", "car", ...]  # Full COCO class list would go here
                self.log_message.emit(f"Detected {coco_classes[best_class]} with confidence {best_score:.2f}")
            
            # Convert mask to proper format for OpenCV
            mask = (best_mask * 255).astype(np.uint8)
            
            # Refine mask with morphological operations
            kernel = np.ones((5,5), np.uint8)
            mask = cv2.morphologyEx(mask, cv2.MORPH_CLOSE, kernel, iterations=2)
            
            # Apply mask to create output
            if transparent_bg:
                # Create transparent background
                result = cv2.cvtColor(image, cv2.COLOR_BGR2BGRA)
                result[:, :, 3] = mask
            else:
                # Create white background
                white_bg = np.ones_like(image) * 255
                foreground = cv2.bitwise_and(image, image, mask=mask)
                bg_mask = cv2.bitwise_not(mask)
                background = cv2.bitwise_and(white_bg, white_bg, mask=bg_mask)
                result = cv2.add(foreground, background)
            
            return result
            
        except Exception as e:
            error_msg = f"Detectron2 processing failed: {str(e)}"
            self.log_message.emit(error_msg)
            self.log_message.emit(traceback.format_exc())
            self.log_message.emit("Falling back to method 5")
            return self.bg_remove_method5(image)
        
    