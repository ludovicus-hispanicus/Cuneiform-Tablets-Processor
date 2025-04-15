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
                result = self.bg_remove_method1(image)
            elif self.method == 2:
                result = self.bg_remove_method2(image)
            elif self.method == 3:
                result = self.bg_remove_method3(image)
            elif self.method == 4:
                block_size = params.get('block_size', 11)
                c_constant = params.get('c_constant', 2)
                result = self.bg_remove_method4(image, block_size, c_constant)
            elif self.method == 5:
                low_threshold = params.get('low_threshold', 30)
                high_threshold = params.get('high_threshold', 100)
                dilation_iterations = params.get('dilation_iterations', 3)
                result = self.bg_remove_method5(image, low_threshold, high_threshold, dilation_iterations)
            elif self.method == 6:
                k = params.get('k_clusters', 2)
                bg_detection_mode = params.get('bg_detection_mode', 'darkest')
                result = self.bg_remove_method6(image, k, bg_detection_mode)
            elif self.method == 7:
                clahe_clip = params.get('clahe_clip', 4.0)
                edge_sensitivity = params.get('edge_sensitivity', 3)
                iterations = params.get('iterations', 10)
                result = self.bg_remove_method7(image, clahe_clip, edge_sensitivity, iterations)
            elif self.method == 8:
                result = self.bg_remove_rembg(image)
            elif self.method == 9:
                result = self.bg_remove_ml(image)    
            elif self.method == 10:
                result = self.bg_remove_otsu_contour(image)
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
    
    def bg_remove_method1(self, image):
        """
        Background removal using adaptive thresholding with preprocessing
        optimized for cuneiform tablets against light backgrounds
        
        Args:
            image (ndarray): Input image
        
        Returns:
            ndarray: Image with background removed (white background)
        """
        # Convert to grayscale
        gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
        
        # Apply CLAHE for local contrast enhancement
        clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8,8))
        gray = clahe.apply(gray)
        
        # Apply bilateral filter to reduce noise while preserving edges
        blurred = cv2.bilateralFilter(gray, 9, 75, 75)
        
        # Adaptive thresholding - works better than global threshold for uneven lighting
        thresh = cv2.adaptiveThreshold(
            blurred, 
            255, 
            cv2.ADAPTIVE_THRESH_GAUSSIAN_C, 
            cv2.THRESH_BINARY_INV, 
            21,  # Larger block size for tablets
            5    # Smaller constant for cleaner separation
        )
        
        # Morphological operations to clean up the mask
        kernel = np.ones((3,3), np.uint8)
        cleaned = cv2.morphologyEx(thresh, cv2.MORPH_OPEN, kernel, iterations=2)
        cleaned = cv2.morphologyEx(cleaned, cv2.MORPH_CLOSE, kernel, iterations=3)
        
        # Find contours to get the largest object (presumably the tablet)
        contours, _ = cv2.findContours(cleaned, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        if contours:
            # Get the largest contour
            largest_contour = max(contours, key=cv2.contourArea)
            
            # Create a mask from the largest contour
            mask = np.zeros_like(gray)
            cv2.drawContours(mask, [largest_contour], -1, 255, thickness=cv2.FILLED)
            
            # Smooth the mask edges
            mask = cv2.GaussianBlur(mask, (5,5), 0)
            _, mask = cv2.threshold(mask, 127, 255, cv2.THRESH_BINARY)
        else:
            # Fallback to the cleaned threshold if no contours found
            mask = cleaned
        
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

    def bg_remove_method2(self, image):
        """
        Improved background remover method 2 using adaptive thresholding
        
        Args:
            image (ndarray): Input image
        
        Returns:
            ndarray: Image with background removed
        """
        # Convert to Grayscale
        image_grey = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
        
        # Apply some blurring to reduce noise
        image_grey = cv2.GaussianBlur(image_grey, (5, 5), 0)

        # Use adaptive thresholding instead of fixed value
        # This works better across different lighting conditions
        mask = cv2.adaptiveThreshold(
            image_grey, 
            255, 
            cv2.ADAPTIVE_THRESH_GAUSSIAN_C, 
            cv2.THRESH_BINARY_INV, 
            11,  # Block size
            2    # Constant subtracted from mean
        )
        
        # Clean up the mask with morphological operations
        kernel = np.ones((3, 3), np.uint8)
        mask = cv2.morphologyEx(mask, cv2.MORPH_OPEN, kernel, iterations=1)
        mask = cv2.morphologyEx(mask, cv2.MORPH_CLOSE, kernel, iterations=2)
        
        # Create white background
        white_bg = np.ones_like(image) * 255
        
        # Apply mask to extract foreground
        foreground = cv2.bitwise_and(image, image, mask=mask)
        
        # Invert mask for background
        bg_mask = cv2.bitwise_not(mask)
        background = cv2.bitwise_and(white_bg, white_bg, mask=bg_mask)
        
        # Combine foreground and background
        final_image = cv2.add(foreground, background)
        
        return final_image

    def bg_remove_method3(self, image):
        """
        Improved background remover method 3 using HSV color space
        
        Args:
            image (ndarray): Input image
        
        Returns:
            ndarray: Image with background removed
        """
        # Convert to HSV color space
        image_hsv = cv2.cvtColor(image, cv2.COLOR_BGR2HSV)
        
        # Get individual channels
        h, s, v = cv2.split(image_hsv)
        
        # Dynamically determine thresholds based on image content
        # Calculate the mean of saturation and value channels
        s_mean = np.mean(s)
        v_mean = np.mean(v)
        
        # Set thresholds relative to the mean values
        s_threshold = max(20, s_mean * 0.5)  # At least 20, or half the mean
        v_threshold = max(20, v_mean * 0.5)  # At least 20, or half the mean
        
        # Create binary masks for saturation and value
        s_mask = np.where(s > s_threshold, 255, 0).astype(np.uint8)
        v_mask = np.where(v > v_threshold, 255, 0).astype(np.uint8)
        
        # Combine masks
        combined_mask = cv2.bitwise_or(s_mask, v_mask)
        
        # Clean up the mask with morphological operations
        kernel = np.ones((5, 5), np.uint8)
        combined_mask = cv2.morphologyEx(combined_mask, cv2.MORPH_CLOSE, kernel, iterations=2)
        combined_mask = cv2.morphologyEx(combined_mask, cv2.MORPH_OPEN, kernel, iterations=1)
        
        # Create white background
        white_bg = np.ones_like(image) * 255
        
        # Apply mask to extract foreground
        foreground = cv2.bitwise_and(image, image, mask=combined_mask)
        
        # Invert mask for background
        bg_mask = cv2.bitwise_not(combined_mask)
        background = cv2.bitwise_and(white_bg, white_bg, mask=bg_mask)
        
        # Combine foreground and background
        final_image = cv2.add(foreground, background)
        
        return final_image
    
    def bg_remove_method4(self, image, block_size=11, c_constant=2):
        """
        Remove black background from an image. Works better when
        objects are photographed against a dark/black background.
        
        Args:
            image (ndarray): Input image with black background
            block_size (int): Block size for adaptive threshold
            c_constant (int): Constant subtracted from mean
        
        Returns:
            ndarray: Image with background replaced by white
        """
        # Method implementation with parameters
        gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
        
        # Use adaptive thresholding to handle varying lighting
        # This works better than global thresholding for uneven backgrounds
        binary = cv2.adaptiveThreshold(
            gray, 
            255, 
            cv2.ADAPTIVE_THRESH_GAUSSIAN_C, 
            cv2.THRESH_BINARY_INV, 
            block_size,  # Now uses parameter
            c_constant   # Now uses parameter
        )
        
        # Clean up the mask - remove noise
        kernel = np.ones((3, 3), np.uint8)
        binary = cv2.morphologyEx(binary, cv2.MORPH_OPEN, kernel, iterations=1)
        binary = cv2.morphologyEx(binary, cv2.MORPH_CLOSE, kernel, iterations=2)
        
        # Create white background
        white_bg = np.ones_like(image) * 255
        
        # Apply mask to extract foreground
        foreground = cv2.bitwise_and(image, image, mask=binary)
        
        # Invert mask for background
        background_mask = cv2.bitwise_not(binary)
        background = cv2.bitwise_and(white_bg, white_bg, mask=background_mask)
        
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

    def bg_remove_method6(self, image, k=2, bg_detection_mode='darkest'):
        """
        Background remover method 6 using k-means clustering
        
        Args:
            image (ndarray): Input image
            k (int): Number of clusters for k-means (default: 2)
            bg_detection_mode (str): How to identify background cluster
                                    'darkest' - assumes darkest cluster is background
                                    'brightest' - assumes brightest cluster is background
                                    'largest' - assumes largest cluster is background
        
        Returns:
            ndarray: Image with background removed
        """
        # Reshape image
        pixels = image.reshape((-1, 3))
        pixels = np.float32(pixels)
        
        # Define criteria and apply kmeans
        criteria = (cv2.TERM_CRITERIA_EPS + cv2.TERM_CRITERIA_MAX_ITER, 100, 0.2)
        _, labels, centers = cv2.kmeans(pixels, k, None, criteria, 10, cv2.KMEANS_RANDOM_CENTERS)
        
        # Convert back to uint8
        centers = np.uint8(centers)
        segmented = centers[labels.flatten()]
        segmented = segmented.reshape(image.shape)
        
        # Find which cluster is the background based on selected mode
        if bg_detection_mode == 'darkest':
            # Find darkest cluster (lowest average intensity)
            avg_colors = np.mean(centers, axis=1)
            bg_cluster = np.argmin(avg_colors)
        elif bg_detection_mode == 'brightest':
            # Find brightest cluster (highest average intensity)
            avg_colors = np.mean(centers, axis=1)
            bg_cluster = np.argmax(avg_colors)
        elif bg_detection_mode == 'largest':
            # Find the largest cluster (most pixels)
            cluster_sizes = np.bincount(labels.flatten())
            bg_cluster = np.argmax(cluster_sizes)
        else:
            # Default to darkest
            avg_colors = np.mean(centers, axis=1)
            bg_cluster = np.argmin(avg_colors)
        
        # Create mask - 0 for background cluster, 255 for others
        mask = np.ones(labels.shape, np.uint8) * 255
        mask[labels.flatten() == bg_cluster] = 0
        mask = mask.reshape(image.shape[0], image.shape[1])
        
        # Create white background
        white_bg = np.ones_like(image) * 255
        
        # Apply mask
        foreground = cv2.bitwise_and(image, image, mask=mask)
        bg_mask = cv2.bitwise_not(mask)
        background = cv2.bitwise_and(white_bg, white_bg, mask=bg_mask)
        
        # Combine
        result = cv2.add(foreground, background)
        
        return result
    
    def bg_remove_method7(self, image, clahe_clip=4.0, edge_sensitivity=3, iterations=10):
        """
        GrabCut optimized specifically for cuneiform tablets against dark backgrounds
        """
        # Create initial mask
        mask = np.zeros(image.shape[:2], np.uint8)
        
        # Map edge sensitivity to appropriate thresholds
        edge_thresholds = {
            1: (30, 120),  # Less sensitive
            2: (25, 110),
            3: (20, 100),  # Default
            4: (15, 75),
            5: (10, 50)    # More sensitive
        }
        low_thresh, high_thresh = edge_thresholds.get(edge_sensitivity, (20, 100))
        
        # Convert to grayscale and enhance contrast based on parameter
        gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
        clahe = cv2.createCLAHE(clipLimit=clahe_clip, tileGridSize=(8,8))
        gray = clahe.apply(gray)
        
        # Try various preprocessing to improve edge detection
        blurred = cv2.GaussianBlur(gray, (5, 5), 0)
        
        # Lower thresholds for Canny to detect more edges based on sensitivity
        edges = cv2.Canny(blurred, low_thresh, high_thresh)
        
        # More aggressive dilation to connect edges
        kernel = np.ones((7,7), np.uint8)  # Larger kernel
        dilated = cv2.dilate(edges, kernel, iterations=4)  # More iterations
        
        # Find contours 
        contours, _ = cv2.findContours(dilated, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        
        # Define variables needed for GrabCut
        bgd_model = np.zeros((1, 65), np.float64)
        fgd_model = np.zeros((1, 65), np.float64)
        
        # Default rectangle in case none is found
        h, w = image.shape[:2]
        rect = (int(w*0.1), int(h*0.1), int(w*0.8), int(h*0.8))
        
        # If no significant contours found, try with even lower thresholds
        if not contours or max((cv2.contourArea(c) for c in contours), default=0) < (image.shape[0] * image.shape[1] * 0.05):
            edges = cv2.Canny(gray, low_thresh/2, high_thresh/2)  # Even lower thresholds
            dilated = cv2.dilate(edges, kernel, iterations=5)
            contours, _ = cv2.findContours(dilated, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        
        # If contours are found, use them to create mask
        if contours and max((cv2.contourArea(c) for c in contours), default=0) > (image.shape[0] * image.shape[1] * 0.01):
            # Find largest contour by area
            largest_contour = max(contours, key=cv2.contourArea)
            
            # Get bounding rectangle with generous margin
            x, y, w, h = cv2.boundingRect(largest_contour)
            margin_x = int(w * 15 / 100)
            margin_y = int(h * 15 / 100)
            x = max(0, x - margin_x)
            y = max(0, y - margin_y)
            w = min(image.shape[1] - x, w + 2*margin_x)
            h = min(image.shape[0] - y, h + 2*margin_y)
            rect = (x, y, w, h)
            
            # More nuanced initialization
            # Definitely foreground: interior of contour
            cv2.drawContours(mask, [largest_contour], 0, cv2.GC_FGD, -1)
            
            # Probable foreground: area around contour
            dilated_contour = cv2.dilate(np.zeros_like(gray), kernel, iterations=2)
            cv2.drawContours(dilated_contour, [largest_contour], 0, 255, -1)
            mask[dilated_contour == 255] = cv2.GC_PR_FGD
            
            # Rest is background
            mask[mask == 0] = cv2.GC_BGD
        else:
            # If no contours or too small, try intensity-based approach
            _, intensity_mask = cv2.threshold(gray, 40, 255, cv2.THRESH_BINARY)
            kernel = np.ones((5,5), np.uint8)
            intensity_mask = cv2.morphologyEx(intensity_mask, cv2.MORPH_CLOSE, kernel, iterations=2)
            
            # Find contours in the intensity mask
            intensity_contours, _ = cv2.findContours(intensity_mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
            
            if intensity_contours and any(cv2.contourArea(c) > (image.shape[0] * image.shape[1] * 0.01) for c in intensity_contours):
                largest_contour = max(intensity_contours, key=cv2.contourArea)
                x, y, w, h = cv2.boundingRect(largest_contour)
                rect = (x, y, w, h)
                
                # Initialize mask: probable foreground for contour interior
                cv2.drawContours(mask, [largest_contour], 0, cv2.GC_PR_FGD, -1)
                
                # Mark rest as background
                mask[mask == 0] = cv2.GC_BGD
            else:
                # Final fallback - use center with larger margin
                h, w = image.shape[:2]
                margin_x = int(w * 25 / 100)  # Larger margin
                margin_y = int(h * 25 / 100)
                rect = (margin_x, margin_y, w - 2*margin_x, h - 2*margin_y)
                
                # Initialize with rectangle
                mask[rect[1]:rect[1]+rect[3], rect[0]:rect[0]+rect[2]] = cv2.GC_PR_FGD
                mask[mask == 0] = cv2.GC_BGD
        
        # Apply GrabCut with more iterations for better results
        try:
            cv2.grabCut(image, mask, rect, bgd_model, fgd_model, iterations, cv2.GC_INIT_WITH_MASK)
        except Exception as e:
            self.log_message.emit(f"GrabCut error: {str(e)}, falling back to simpler method")
            # Fallback
            mask = np.zeros(image.shape[:2], np.uint8)
            mask[rect[1]:rect[1]+rect[3], rect[0]:rect[0]+rect[2]] = 1
        
        # Create mask (both definite and probable foreground)
        mask2 = np.where((mask == 2) | (mask == 0), 0, 1).astype('uint8')
        
        # Refine edges with morphological operations
        kernel = np.ones((3, 3), np.uint8)
        mask2 = cv2.morphologyEx(mask2, cv2.MORPH_CLOSE, kernel, iterations=3)
        
        # Sometimes artifacts remain at the edges - clean them up
        mask2 = cv2.morphologyEx(mask2, cv2.MORPH_OPEN, kernel, iterations=1)
        
        # Create output image
        foreground = cv2.bitwise_and(image, image, mask=mask2)
        background = np.ones_like(image, np.uint8) * 255
        background = cv2.bitwise_and(background, background, mask=cv2.bitwise_not(mask2))
        result = cv2.add(foreground, background)
        
        return result           
    def bg_remove_rembg(self, image):
        """
        Background remover using Rembg library with U²-Net
        Excellent for objects against uniform backgrounds
        
        Args:
            image (ndarray): Input image (BGR format from OpenCV)
        
        Returns:
            ndarray: Image with background removed (white background)
        """
        try:
            # Import rembg (you'll need to install with pip install rembg)
            from rembg import remove
            import PIL.Image as PILImage
            from io import BytesIO
            
            # Convert OpenCV image (BGR) to PIL Image (RGB)
            image_rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
            pil_image = PILImage.fromarray(image_rgb)
            
            # Process with rembg
            output_pil = remove(pil_image)
            
            # Extract alpha channel and create white background
            alpha = output_pil.split()[-1]
            bg = PILImage.new("RGB", output_pil.size, (255, 255, 255))
            bg.paste(output_pil, mask=alpha)
            
            # Convert back to OpenCV format
            result = cv2.cvtColor(np.array(bg), cv2.COLOR_RGB2BGR)
            
            return result
        except ImportError:
            self.log_message.emit("Rembg library not installed. Install with 'pip install rembg'")
            # Fall back to method 6 which works well for this type of image
            return self.bg_remove_method6(image, k=2, bg_detection_mode='darkest')
    
    def bg_remove_ml(self, image):
        """
        Background remover using a pre-trained U-Net or DeepLabV3 model
        specifically fine-tuned for archaeological artifacts
        
        Args:
            image (ndarray): Input image
        
        Returns:
            ndarray: Image with background removed
        """
        try:
            # Import necessary libraries
            import torch
            from torchvision import transforms
            
            # Check if we can use GPU
            device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
            
            # Load pre-trained model (you would need to train this or find one)
            # This is a placeholder - you'd need to implement the actual model loading
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
            input_batch = input_tensor.unsqueeze(0).to(device)
            
            # Get prediction
            with torch.no_grad():
                output = model(input_batch)['out'][0]
            output_predictions = output.argmax(0).byte().cpu().numpy()
            
            # Resize mask back to original image size
            mask = cv2.resize(output_predictions, (image.shape[1], image.shape[0]), 
                              interpolation=cv2.INTER_NEAREST)
            
            # Create foreground mask (assuming class 15 is the object class for DeepLabV3)
            # You would need to adjust this based on your model's output
            foreground_mask = (mask == 15).astype(np.uint8) * 255
            
            # Create white background
            white_bg = np.ones_like(image) * 255
            
            # Apply mask to extract foreground
            foreground = cv2.bitwise_and(image, image, mask=foreground_mask)
            
            # Invert mask for background
            bg_mask = cv2.bitwise_not(foreground_mask)
            background = cv2.bitwise_and(white_bg, white_bg, mask=bg_mask)
            
            # Combine foreground and background
            result = cv2.add(foreground, background)
            
            return result
        
        except ImportError:
            self.log_message.emit("Required ML libraries not installed. Falling back to Otsu+Contour method.")
            return self.bg_remove_otsu_contour(image)
        
    def bg_remove_otsu_contour(self, image):
        """
        Background remover optimized for clay tablets against dark backgrounds
        using Otsu thresholding combined with contour refinement
        
        Args:
            image (ndarray): Input image
        
        Returns:
            ndarray: Image with background removed
        """
        # Convert to grayscale
        gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
        
        # Apply bilateral filter to smooth the image while preserving edges
        blurred = cv2.bilateralFilter(gray, 9, 75, 75)
        
        # Apply Otsu's thresholding
        _, thresh = cv2.threshold(blurred, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
        
        # Morphological operations to clean up the mask
        kernel = np.ones((5, 5), np.uint8)
        opening = cv2.morphologyEx(thresh, cv2.MORPH_OPEN, kernel, iterations=2)
        sure_bg = cv2.dilate(opening, kernel, iterations=3)
        
        # Distance transform to find sure foreground
        dist_transform = cv2.distanceTransform(opening, cv2.DIST_L2, 5)
        _, sure_fg = cv2.threshold(dist_transform, 0.3*dist_transform.max(), 255, 0)
        sure_fg = sure_fg.astype(np.uint8)
        
        # Find unknown region
        unknown = cv2.subtract(sure_bg, sure_fg)
        
        # Marker labelling
        _, markers = cv2.connectedComponents(sure_fg)
        markers = markers + 1
        markers[unknown==255] = 0
        
        # Apply watershed algorithm
        markers = cv2.watershed(image, markers)
        
        # Create mask - mark region of interest with white
        mask = np.zeros_like(gray, dtype=np.uint8)
        mask[markers > 1] = 255
        
        # Find contours on the mask
        contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        
        # Select the largest contour
        if contours:
            largest_contour = max(contours, key=cv2.contourArea)
            
            # Create a new mask with only the largest contour
            refined_mask = np.zeros_like(gray, dtype=np.uint8)
            cv2.drawContours(refined_mask, [largest_contour], 0, 255, -1)
            
            # Apply a final dilation to ensure we capture the entire object
            refined_mask = cv2.dilate(refined_mask, kernel, iterations=2)
        else:
            refined_mask = mask
        
        # Create white background
        white_bg = np.ones_like(image) * 255
        
        # Apply mask to extract foreground
        foreground = cv2.bitwise_and(image, image, mask=refined_mask)
        
        # Invert mask for background
        bg_mask = cv2.bitwise_not(refined_mask)
        background = cv2.bitwise_and(white_bg, white_bg, mask=bg_mask)
        
        # Combine foreground and background
        result = cv2.add(foreground, background)
        
        return result
        
        # def create_transparent_background(self, input_path, output_path):
        #     try:
        #         # Read the image
        #         image = cv2.imread(input_path)
                
        #         # Get alpha channel (foreground mask)
        #         if self.method == 1:
        #             result = self.bg_remove_method1(image)
        #         elif self.method == 2:
        #             result = self.bg_remove_method2(image)
        #         elif self.method == 3:
        #             result = self.bg_remove_method3(image)
        #         elif self.method == 4:
        #             result = self.bg_remove_method4(image)
        #         elif self.method == 5:
        #             result = self.bg_remove_method5(image)
        #         elif self.method == 6:
        #             result = self.bg_remove_method6(image)
        #         elif self.method == 7:
        #             result = self.bg_remove_method7(image)
        #         elif self.method == 8:
        #         # Neural network-based approach
        #             result = self.bg_remove_rembg(image)
        #         else:
        #             result = self.bg_remove_method2(image)  # Default to method 2
                
        #         # Convert to grayscale and threshold
        #         gray = cv2.cvtColor(result, cv2.COLOR_BGR2GRAY)
        #         _, alpha = cv2.threshold(gray, 250, 255, cv2.THRESH_BINARY_INV)
                
        #         # Convert to PIL Image
        #         b, g, r = cv2.split(image)
        #         img_rgba = Image.merge('RGBA', (
        #             Image.fromarray(r),
        #             Image.fromarray(g),
        #             Image.fromarray(b),
        #             Image.fromarray(alpha)
        #         ))
                
        #         # Save with transparency
        #         img_rgba.save(output_path, format='PNG')
        #         self.log_message.emit(f"Saved transparent background image: {output_path}")
                
        #         return True
            
        #     except Exception as e:
        #         self.log_message.emit(f"Error creating transparent background: {str(e)}")
        #         tb = traceback.format_exc()
        #         self.log_message.emit(f"Traceback: {tb}")
        #         return False