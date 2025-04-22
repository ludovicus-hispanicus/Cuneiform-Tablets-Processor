#!/usr/bin/env python3
"""
Main window implementation for the Cuneiform Tablet Processor application.
"""

import os
import sys
import traceback
import cv2
import numpy as np
import torch

# Update imports at the top of the main_window.py file:
from PyQt5.QtWidgets import (QMainWindow, QTabWidget, QWidget, 
                           QVBoxLayout, QHBoxLayout, QLabel, 
                           QLineEdit, QPushButton, QFileDialog,
                           QCheckBox, QRadioButton, QButtonGroup, 
                           QSpinBox, QDoubleSpinBox, QComboBox, QProgressBar, QMessageBox,
                           QGroupBox, QSlider, QTextEdit, QPlainTextEdit,
                           QDockWidget, QFormLayout)
from PyQt5.QtCore import Qt, QSettings, pyqtSignal
from PyQt5.QtGui import QPixmap, QIcon, QTextCursor, QColor
from PyQt5.QtCore import QTimer
import tempfile

# Try to import RawProcessor, but don't fail if it's not available
try:
    from processing.raw_processor import RawProcessor
    RAW_PROCESSING_AVAILABLE = True
except ImportError:
    RAW_PROCESSING_AVAILABLE = False
    
try:
    from processing.background_remover import BackgroundRemover
    BACKGROUND_REMOVAL_AVAILABLE = True
except ImportError:
    BACKGROUND_REMOVAL_AVAILABLE = False

from processing.processor import ImageProcessor

class MainWindow(QMainWindow):
    """Main application window with tabbed interface"""
    def __init__(self):
        super().__init__()
        self.setWindowTitle("Cuneiform Tablet Processor")
        self.setMinimumSize(800, 700)
        
        # Load settings
        self.settings = QSettings("TabletProcessor", "CuneiformApp")
        
        # Create main widget and layout
        self.central_widget = QWidget()
        self.setCentralWidget(self.central_widget)
        self.main_layout = QVBoxLayout(self.central_widget)
        
        # Create tabbed interface
        self.tabs = QTabWidget()
        self.main_layout.addWidget(self.tabs)
        
        # Create tabs
        self.create_processing_tab()
        self.create_metadata_tab()
        
        # Check if RAW processing is available and add the tab if it is
        if RAW_PROCESSING_AVAILABLE:
            self.create_raw_processing_tab()
        
        # Check if background removal is available and add the tab if it is
        if BACKGROUND_REMOVAL_AVAILABLE:
            self.create_bg_removal_tab()
        
        # Create buttons
        self.create_buttons()
        
        # Load saved settings
        self.load_settings()
        
        # Initialize processors
        self.processor = None
        self.raw_processor = None
        self.bg_remover = None
    
    
    
    def create_processing_tab(self):
        """Create Processing Options tab"""
        tab = QWidget()
        layout = QVBoxLayout(tab)
        
        # Folders Panel
        folders_group = QGroupBox("Source and Destination")
        folders_layout = QVBoxLayout(folders_group)
        
        # Source Folder
        source_layout = QHBoxLayout()
        source_layout.addWidget(QLabel("Source Folder:"))
        self.source_input = QLineEdit()
        source_layout.addWidget(self.source_input)
        self.source_browse = QPushButton("Browse")
        self.source_browse.clicked.connect(self.browse_source)
        source_layout.addWidget(self.source_browse)
        folders_layout.addLayout(source_layout)
        
        # Output Folder
        output_layout = QHBoxLayout()
        output_layout.addWidget(QLabel("Output Folder:"))
        self.output_input = QLineEdit()
        output_layout.addWidget(self.output_input)
        self.output_browse = QPushButton("Browse")
        self.output_browse.clicked.connect(self.browse_output)
        output_layout.addWidget(self.output_browse)
        folders_layout.addLayout(output_layout)
        
        layout.addWidget(folders_group)
        
        # Logo Options
        logo_group = QGroupBox("Logo Options")
        logo_layout = QVBoxLayout(logo_group)
        
        # Add Logo Checkbox
        logo_check_layout = QHBoxLayout()
        self.add_logo_checkbox = QCheckBox("Add Logo to Images")
        self.add_logo_checkbox.stateChanged.connect(self.toggle_logo_controls)
        logo_check_layout.addWidget(self.add_logo_checkbox)
        logo_layout.addLayout(logo_check_layout)
        
        # Logo Path
        logo_path_layout = QHBoxLayout()
        logo_path_layout.addWidget(QLabel("Logo File:"))
        self.logo_path_input = QLineEdit()
        logo_path_layout.addWidget(self.logo_path_input)
        self.logo_browse = QPushButton("Browse")
        self.logo_browse.clicked.connect(self.browse_logo)
        logo_path_layout.addWidget(self.logo_browse)
        logo_layout.addLayout(logo_path_layout)
        
        layout.addWidget(logo_group)
        
        # TIFF Options
        tiff_group = QGroupBox("TIFF Options")
        tiff_layout = QVBoxLayout(tiff_group)
        
        # DPI Selection
        dpi_layout = QHBoxLayout()
        dpi_layout.addWidget(QLabel("Final Resolution:"))
        dpi_radio_group = QHBoxLayout()
        self.dpi_300_radio = QRadioButton("300 DPI")
        self.dpi_600_radio = QRadioButton("600 DPI")
        dpi_radio_group.addWidget(self.dpi_300_radio)
        dpi_radio_group.addWidget(self.dpi_600_radio)
        dpi_layout.addLayout(dpi_radio_group)
        tiff_layout.addLayout(dpi_layout)
        
        # TIFF Compression
        compression_layout = QHBoxLayout()
        compression_layout.addWidget(QLabel("Compression:"))
        self.compression_dropdown = QComboBox()
        self.compression_dropdown.addItems(["None", "LZW", "ZIP"])
        compression_layout.addWidget(self.compression_dropdown)
        tiff_layout.addLayout(compression_layout)
        
        layout.addWidget(tiff_group)
        
        # JPEG Options
        jpeg_group = QGroupBox("JPEG Options")
        jpeg_layout = QVBoxLayout(jpeg_group)
        
        # Save JPEG Checkbox
        jpeg_check_layout = QHBoxLayout()
        self.save_jpeg_checkbox = QCheckBox("Save JPEG Copy")
        self.save_jpeg_checkbox.stateChanged.connect(self.toggle_jpeg_controls)
        jpeg_check_layout.addWidget(self.save_jpeg_checkbox)
        jpeg_layout.addLayout(jpeg_check_layout)
        
        # JPEG Quality
        quality_header_layout = QHBoxLayout()
        quality_header_layout.addWidget(QLabel("JPEG Quality:"))
        self.quality_level_text = QLabel("")
        self.quality_level_text.setMinimumWidth(100)
        quality_header_layout.addWidget(self.quality_level_text)
        jpeg_layout.addLayout(quality_header_layout)
        
        quality_control_layout = QHBoxLayout()
        self.quality_dropdown = QComboBox()
        self.quality_dropdown.addItems([str(i) for i in range(1, 13)])
        self.quality_dropdown.setCurrentIndex(7)  # Default to 8
        self.quality_dropdown.currentIndexChanged.connect(self.sync_quality_controls)
        quality_control_layout.addWidget(self.quality_dropdown)
        
        self.quality_slider = QSlider(Qt.Horizontal)
        self.quality_slider.setMinimum(1)
        self.quality_slider.setMaximum(12)
        self.quality_slider.setValue(8)
        self.quality_slider.setTickPosition(QSlider.TicksBelow)
        self.quality_slider.setTickInterval(1)
        self.quality_slider.valueChanged.connect(self.sync_quality_controls)
        quality_control_layout.addWidget(self.quality_slider)
        
        self.quality_value = QLabel("8")
        self.quality_value.setMinimumWidth(20)
        quality_control_layout.addWidget(self.quality_value)
        
        jpeg_layout.addLayout(quality_control_layout)
        layout.addWidget(jpeg_group)
        
        # Update the quality level text
        self.update_quality_level_text(8)
        
        self.tabs.addTab(tab, "Processing Options")
    
    def create_metadata_tab(self):
        """Create Metadata tab"""
        tab = QWidget()
        layout = QVBoxLayout(tab)
        
        # General Metadata
        general_group = QGroupBox("General Metadata")
        general_layout = QVBoxLayout(general_group)
        
        # Photographer
        photographer_layout = QHBoxLayout()
        photographer_layout.addWidget(QLabel("Photographer:"))
        self.photographer_input = QLineEdit()
        photographer_layout.addWidget(self.photographer_input)
        general_layout.addLayout(photographer_layout)
        
        # Institution
        institution_layout = QHBoxLayout()
        institution_layout.addWidget(QLabel("Institution:"))
        self.institution_input = QLineEdit()
        institution_layout.addWidget(self.institution_input)
        general_layout.addLayout(institution_layout)
        
        # Copyright Notice
        copyright_layout = QHBoxLayout()
        copyright_layout.addWidget(QLabel("Copyright Notice:"))
        self.copyright_input = QLineEdit()
        copyright_layout.addWidget(self.copyright_input)
        general_layout.addLayout(copyright_layout)
        
        # Usage Terms
        usage_layout = QHBoxLayout()
        usage_layout.addWidget(QLabel("Copyright (Usage Terms):"))
        self.usage_terms_input = QLineEdit()
        usage_layout.addWidget(self.usage_terms_input)
        general_layout.addLayout(usage_layout)
        
        layout.addWidget(general_group)
        
        # IPTC Metadata
        iptc_group = QGroupBox("IPTC Metadata")
        iptc_layout = QVBoxLayout(iptc_group)
        
        # Title - read only, will be set automatically
        title_layout = QHBoxLayout()
        title_layout.addWidget(QLabel("Title:"))
        title_display = QLabel("(Will be set to Tablet Number)")
        title_layout.addWidget(title_display)
        iptc_layout.addLayout(title_layout)
        
        # Headline - read only, will be set automatically
        headline_layout = QHBoxLayout()
        headline_layout.addWidget(QLabel("Headline:"))
        headline_display = QLabel("(Will be set to Tablet Number)")
        headline_layout.addWidget(headline_display)
        iptc_layout.addLayout(headline_layout)
        
        # Author/Creator - linked to photographer field
        author_layout = QHBoxLayout()
        author_layout.addWidget(QLabel("Author/Creator:"))
        self.author_display = QLabel("")
        author_layout.addWidget(self.author_display)
        iptc_layout.addLayout(author_layout)
        
        # Credit Line
        credit_layout = QHBoxLayout()
        credit_layout.addWidget(QLabel("Credit Line:"))
        credit_layout.addStretch(1)
        iptc_layout.addLayout(credit_layout)
        
        self.credit_input = QTextEdit()
        default_credit = (
            ""
        )
        self.credit_input.setText(default_credit)
        self.credit_input.setMaximumHeight(100)
        iptc_layout.addWidget(self.credit_input)
        
        layout.addWidget(iptc_group)
        
        self.tabs.addTab(tab, "Metadata")
        
        # Connect photographer field to author display
        self.photographer_input.textChanged.connect(self.update_author_display)
    
    def create_buttons(self):
        """Create action buttons and progress bar"""
        button_layout = QHBoxLayout()
        
        self.process_button = QPushButton("Process Images")
        self.process_button.clicked.connect(self.process_images)
        
        self.cancel_button = QPushButton("Cancel")
        self.cancel_button.clicked.connect(self.close)
        
        button_layout.addWidget(self.process_button)
        button_layout.addWidget(self.cancel_button)
        
        self.main_layout.addLayout(button_layout)
        
        # Add progress bar
        self.progress_bar = QProgressBar()
        self.progress_bar.setVisible(False)
        self.main_layout.addWidget(self.progress_bar)
        
        # Add log area
        self.create_log_area()
    
    def load_settings(self):
        """Load saved settings"""
        self.source_input.setText(self.settings.value("source_path", ""))
        self.output_input.setText(self.settings.value("output_path", ""))
        
        self.add_logo_checkbox.setChecked(self.settings.value("add_logo", False, type=bool))
        self.logo_path_input.setText(self.settings.value("logo_path", ""))
        
        dpi = self.settings.value("dpi", 600, type=int)
        if dpi == 300:
            self.dpi_300_radio.setChecked(True)
        else:
            self.dpi_600_radio.setChecked(True)
        
        compression = self.settings.value("compression", "none")
        index = self.compression_dropdown.findText(compression.capitalize())
        if index >= 0:
            self.compression_dropdown.setCurrentIndex(index)
        
        self.save_jpeg_checkbox.setChecked(self.settings.value("save_jpeg", False, type=bool))
        
        jpeg_quality = self.settings.value("jpeg_quality", 8, type=int)
        self.quality_slider.setValue(jpeg_quality)
        self.quality_dropdown.setCurrentIndex(jpeg_quality - 1)
        self.quality_value.setText(str(jpeg_quality))
        self.update_quality_level_text(jpeg_quality)
        
        self.photographer_input.setText(self.settings.value("photographer", ""))
        self.institution_input.setText(self.settings.value("institution", ""))
        self.copyright_input.setText(self.settings.value("copyright_notice", ""))
        self.usage_terms_input.setText(self.settings.value("usage_terms", ""))
        
        saved_credit = self.settings.value("credit_line", "")
        if saved_credit:
            self.credit_input.setText(saved_credit)
        
        # Update enabled/disabled state of controls
        self.toggle_logo_controls()
        self.toggle_jpeg_controls()
    
    def save_settings(self):
        """Save current settings"""
        self.settings.setValue("source_path", self.source_input.text())
        self.settings.setValue("output_path", self.output_input.text())
        
        self.settings.setValue("add_logo", self.add_logo_checkbox.isChecked())
        self.settings.setValue("logo_path", self.logo_path_input.text())
        
        self.settings.setValue("dpi", 300 if self.dpi_300_radio.isChecked() else 600)
        self.settings.setValue("compression", self.compression_dropdown.currentText().lower())
        
        self.settings.setValue("save_jpeg", self.save_jpeg_checkbox.isChecked())
        self.settings.setValue("jpeg_quality", self.quality_slider.value())
        
        self.settings.setValue("photographer", self.photographer_input.text())
        self.settings.setValue("institution", self.institution_input.text())
        self.settings.setValue("copyright_notice", self.copyright_input.text())
        self.settings.setValue("usage_terms", self.usage_terms_input.text())
        self.settings.setValue("credit_line", self.credit_input.toPlainText())
        
        # Save RAW processing settings if the tab exists
        if RAW_PROCESSING_AVAILABLE and hasattr(self, 'raw_source_input'):
            self.save_raw_settings()
        
        # Save Background Removal settings if the tab exists
        if BACKGROUND_REMOVAL_AVAILABLE and hasattr(self, 'bg_source_input'):
            self.save_bg_settings()
    
    def browse_source(self):
        """Browse for source folder"""
        folder = QFileDialog.getExistingDirectory(self, "Select Source Folder", self.source_input.text())
        if folder:
            self.source_input.setText(folder)
    
    def browse_output(self):
        """Browse for output folder"""
        folder = QFileDialog.getExistingDirectory(self, "Select Output Folder", self.output_input.text())
        if folder:
            self.output_input.setText(folder)
    
    def browse_logo(self):
        """Browse for logo file"""
        file, _ = QFileDialog.getOpenFileName(
            self, 
            "Select Logo File", 
            self.logo_path_input.text(),
            "Image Files (*.png *.jpg *.jpeg *.tif *.tiff *.psd)"
        )
        if file:
            self.logo_path_input.setText(file)
    
    def toggle_logo_controls(self):
        """Enable/disable logo path controls based on checkbox"""
        enabled = self.add_logo_checkbox.isChecked()
        self.logo_path_input.setEnabled(enabled)
        self.logo_browse.setEnabled(enabled)
    
    def toggle_jpeg_controls(self):
        """Enable/disable JPEG quality controls based on checkbox"""
        enabled = self.save_jpeg_checkbox.isChecked()
        self.quality_dropdown.setEnabled(enabled)
        self.quality_slider.setEnabled(enabled)
        self.quality_value.setEnabled(enabled)
        self.quality_level_text.setEnabled(enabled)
    
    def update_quality_level_text(self, value):
        """Update quality level text based on slider value"""
        if value <= 4:
            self.quality_level_text.setText("(Minimum)")
        elif value <= 7:
            self.quality_level_text.setText("(Low)")
        elif value <= 9:
            self.quality_level_text.setText("(Medium)")
        else:
            self.quality_level_text.setText("(Maximum)")
    
    def sync_quality_controls(self, value):
        """Synchronize quality controls when one changes"""
        if self.sender() == self.quality_slider:
            value = self.quality_slider.value()
            self.quality_dropdown.setCurrentIndex(value - 1)
        else:  # Dropdown changed
            value = self.quality_dropdown.currentIndex() + 1
            self.quality_slider.setValue(value)
        
        self.quality_value.setText(str(value))
        self.update_quality_level_text(value)
    
    def update_author_display(self):
        """Update author display when photographer changes"""
        self.author_display.setText(self.photographer_input.text())
    
    def process_images(self):
        """Start image processing"""
        # Clear previous log
        self.clear_log()
        self.log_message("Starting processing...")
        
        # Validate source and destination folders
        source_path = self.source_input.text()
        output_path = self.output_input.text()
        
        if not source_path:
            QMessageBox.warning(self, "Warning", "Please select a source folder.")
            self.log_message("Error: No source folder selected", error=True)
            return
            
        if not os.path.exists(source_path):
            QMessageBox.warning(self, "Warning", "Source folder does not exist.")
            self.log_message(f"Error: Source folder does not exist: {source_path}", error=True)
            return
            
        if not output_path:
            QMessageBox.warning(self, "Warning", "Please select an output folder.")
            self.log_message("Error: No output folder selected", error=True)
            return
            
        if not os.path.exists(output_path):
            response = QMessageBox.question(
                self, 
                "Create Folder", 
                f"Output folder does not exist. Create it?",
                QMessageBox.Yes | QMessageBox.No
            )
            if response == QMessageBox.Yes:
                try:
                    os.makedirs(output_path)
                    self.log_message(f"Created output folder: {output_path}")
                except Exception as e:
                    error_msg = f"Could not create folder: {str(e)}"
                    QMessageBox.critical(self, "Error", error_msg)
                    self.log_message(f"Error: {error_msg}", error=True)
                    return
            else:
                self.log_message("Processing cancelled: Output folder not created", error=True)
                return
                
        # Save settings before processing
        self.save_settings()
        self.log_message("Settings saved")
        
        # Collect settings into a dictionary
        settings_dict = {
            'source_path': source_path,
            'output_path': output_path,
            'add_logo': self.add_logo_checkbox.isChecked(),
            'logo_path': self.logo_path_input.text(),
            'dpi': 300 if self.dpi_300_radio.isChecked() else 600,
            'compression': self.compression_dropdown.currentText().lower(),
            'save_jpeg': self.save_jpeg_checkbox.isChecked(),
            'jpeg_quality': self.quality_slider.value(),
            'photographer': self.photographer_input.text(),
            'institution': self.institution_input.text(),
            'copyright_notice': self.copyright_input.text(),
            'usage_terms': self.usage_terms_input.text(),
            'credit_line': self.credit_input.toPlainText()
        }
        
        # Log important settings
        self.log_message(f"Source path: {source_path}")
        self.log_message(f"Output path: {output_path}")
        self.log_message(f"DPI: {settings_dict['dpi']}")
        self.log_message(f"Compression: {settings_dict['compression']}")
        self.log_message(f"Save JPEG: {settings_dict['save_jpeg']}")
        if settings_dict['add_logo']:
            self.log_message(f"Logo: {settings_dict['logo_path']}")
        
        # Show progress bar
        self.progress_bar.setVisible(True)
        self.progress_bar.setValue(0)
        
        # Disable interface during processing
        self.setEnabled(False)
        
        # Start processing in a separate thread
        self.processor = ImageProcessor(settings_dict)
        self.processor.progress_update.connect(self.update_progress)
        self.processor.processing_complete.connect(self.processing_finished)
        self.processor.error_occurred.connect(self.handle_error)
        self.processor.log_message.connect(self.log_message)
        self.processor.start()
    
    def handle_error(self, error_message):
        """Handle errors from the processor"""
        self.log_message(f"ERROR: {error_message}", error=True)
        self.setEnabled(True)
    
    def update_progress(self, value):
        """Update progress bar"""
        self.progress_bar.setValue(value)
    
    def create_log_area(self):
        """Create a dockable log area"""
        self.log_dock = QDockWidget("Processing Log", self)
        self.log_dock.setAllowedAreas(Qt.BottomDockWidgetArea)
        
        log_widget = QWidget()
        log_layout = QVBoxLayout(log_widget)
        
        self.log_text = QPlainTextEdit()
        self.log_text.setReadOnly(True)
        self.log_text.setMaximumBlockCount(1000)  # Limit to prevent memory issues
        
        log_layout.addWidget(self.log_text)
        
        self.log_dock.setWidget(log_widget)
        self.addDockWidget(Qt.BottomDockWidgetArea, self.log_dock)
        
        # Button to clear log
        log_controls = QHBoxLayout()
        clear_button = QPushButton("Clear Log")
        clear_button.clicked.connect(self.clear_log)
        log_controls.addStretch()
        log_controls.addWidget(clear_button)
        log_layout.addLayout(log_controls)
    
    def log_message(self, message, error=False):
        """Add a message to the log area"""
        self.log_text.moveCursor(QTextCursor.End)
        if error:
            self.log_text.appendHtml(f'<span style="color: red">{message}</span>')
        else:
            self.log_text.appendPlainText(message)
        self.log_text.ensureCursorVisible()
    
    def clear_log(self):
        """Clear the log area"""
        self.log_text.clear()

    def processing_finished(self, success):
        """Handle completion of processing"""
        self.setEnabled(True)
        self.progress_bar.setVisible(False)
        
        if success:
            QMessageBox.information(self, "Complete", "Image processing completed successfully!")
        else:
            QMessageBox.warning(self, "Error", "An error occurred during processing. Check the log for details.")
            
    def create_raw_processing_tab(self):
        """Create RAW Processing tab"""
        tab = QWidget()
        layout = QVBoxLayout(tab)
        
        # Folders Panel
        folders_group = QGroupBox("Source and Destination")
        folders_layout = QVBoxLayout(folders_group)
        
        # Source Folder - reuse the same source input as the processing tab
        source_layout = QHBoxLayout()
        source_layout.addWidget(QLabel("Source Folder:"))
        self.raw_source_input = QLineEdit()
        source_layout.addWidget(self.raw_source_input)
        self.raw_source_browse = QPushButton("Browse")
        self.raw_source_browse.clicked.connect(self.browse_raw_source)
        source_layout.addWidget(self.raw_source_browse)
        folders_layout.addLayout(source_layout)
        
        # Output Folder - reuse the same output input as the processing tab
        output_layout = QHBoxLayout()
        output_layout.addWidget(QLabel("Output Folder:"))
        self.raw_output_input = QLineEdit()
        output_layout.addWidget(self.raw_output_input)
        self.raw_output_browse = QPushButton("Browse")
        self.raw_output_browse.clicked.connect(self.browse_raw_output)
        output_layout.addWidget(self.raw_output_browse)
        folders_layout.addLayout(output_layout)
        
        layout.addWidget(folders_group)
        
        # RAW Processing Options
        raw_options_group = QGroupBox("RAW Processing Options")
        raw_options_layout = QVBoxLayout(raw_options_group)
        
        # White Balance Options
        wb_layout = QHBoxLayout()
        wb_layout.addWidget(QLabel("White Balance:"))
        self.wb_dropdown = QComboBox()
        self.wb_dropdown.addItems(["Camera", "Auto", "None"])
        wb_layout.addWidget(self.wb_dropdown)
        raw_options_layout.addLayout(wb_layout)
        
        # Output Bit Depth
        bit_depth_layout = QHBoxLayout()
        bit_depth_layout.addWidget(QLabel("Output Bit Depth:"))
        self.bit_depth_dropdown = QComboBox()
        self.bit_depth_dropdown.addItems(["16-bit", "8-bit"])
        bit_depth_layout.addWidget(self.bit_depth_dropdown)
        raw_options_layout.addLayout(bit_depth_layout)
        
        # DPI Setting (create new radio buttons for this tab)
        dpi_layout = QHBoxLayout()
        dpi_layout.addWidget(QLabel("Final Resolution:"))
        raw_dpi_radio_group = QButtonGroup(self)
        self.raw_dpi_300_radio = QRadioButton("300 DPI")
        self.raw_dpi_600_radio = QRadioButton("600 DPI")
        self.raw_dpi_600_radio.setChecked(True)  # Default to 600 DPI
        raw_dpi_radio_group.addButton(self.raw_dpi_300_radio)
        raw_dpi_radio_group.addButton(self.raw_dpi_600_radio)
        dpi_radio_hbox = QHBoxLayout()
        dpi_radio_hbox.addWidget(self.raw_dpi_300_radio)
        dpi_radio_hbox.addWidget(self.raw_dpi_600_radio)
        dpi_layout.addLayout(dpi_radio_hbox)
        raw_options_layout.addLayout(dpi_layout)
        
        # TIFF Compression
        compression_layout = QHBoxLayout()
        compression_layout.addWidget(QLabel("Compression:"))
        self.raw_compression_dropdown = QComboBox()
        self.raw_compression_dropdown.addItems(["None", "LZW", "ZIP"])
        compression_layout.addWidget(self.raw_compression_dropdown)
        raw_options_layout.addLayout(compression_layout)
        
        layout.addWidget(raw_options_group)
        
        # Processing Button
        button_layout = QHBoxLayout()
        self.process_raw_button = QPushButton("Convert RAW Files to TIFF")
        self.process_raw_button.clicked.connect(self.process_raw_files)
        button_layout.addWidget(self.process_raw_button)
        
        layout.addLayout(button_layout)
        
        # Add stretch at the bottom
        layout.addStretch()
        
        # Add the tab
        self.tabs.addTab(tab, "RAW Processing")
        
        # Load saved settings for RAW tab
        self.load_raw_settings()

    def load_raw_settings(self):
        """Load saved settings for RAW processing tab"""
        self.raw_source_input.setText(self.settings.value("raw_source_path", ""))
        self.raw_output_input.setText(self.settings.value("raw_output_path", ""))
        
        # White balance
        wb_value = self.settings.value("raw_white_balance", "Camera")
        index = self.wb_dropdown.findText(wb_value)
        if index >= 0:
            self.wb_dropdown.setCurrentIndex(index)
        
        # Bit depth
        bit_depth = self.settings.value("raw_bit_depth", "16-bit")
        index = self.bit_depth_dropdown.findText(bit_depth)
        if index >= 0:
            self.bit_depth_dropdown.setCurrentIndex(index)
        
        # DPI
        raw_dpi = self.settings.value("raw_dpi", 600, type=int)
        if raw_dpi == 300:
            self.raw_dpi_300_radio.setChecked(True)
        else:
            self.raw_dpi_600_radio.setChecked(True)
        
        # Compression
        raw_compression = self.settings.value("raw_compression", "None")
        index = self.raw_compression_dropdown.findText(raw_compression)
        if index >= 0:
            self.raw_compression_dropdown.setCurrentIndex(index)

    def save_raw_settings(self):
        """Save RAW processing settings"""
        self.settings.setValue("raw_source_path", self.raw_source_input.text())
        self.settings.setValue("raw_output_path", self.raw_output_input.text())
        self.settings.setValue("raw_white_balance", self.wb_dropdown.currentText())
        self.settings.setValue("raw_bit_depth", self.bit_depth_dropdown.currentText())
        self.settings.setValue("raw_dpi", 300 if self.raw_dpi_300_radio.isChecked() else 600)
        self.settings.setValue("raw_compression", self.raw_compression_dropdown.currentText())

    def browse_raw_source(self):
        """Browse for RAW source folder"""
        folder = QFileDialog.getExistingDirectory(self, "Select Source Folder with RAW Files", self.raw_source_input.text())
        if folder:
            self.raw_source_input.setText(folder)

    def browse_raw_output(self):
        """Browse for RAW output folder"""
        folder = QFileDialog.getExistingDirectory(self, "Select Output Folder for TIFF Files", self.raw_output_input.text())
        if folder:
            self.raw_output_input.setText(folder)

    def process_raw_files(self):
        """Start RAW file processing"""
        # Clear previous log
        self.clear_log()
        self.log_message("Starting RAW processing...")
        
        # Validate source and destination folders
        source_path = self.raw_source_input.text()
        output_path = self.raw_output_input.text()
        
        if not source_path:
            QMessageBox.warning(self, "Warning", "Please select a source folder.")
            self.log_message("Error: No source folder selected", error=True)
            return
            
        if not os.path.exists(source_path):
            QMessageBox.warning(self, "Warning", "Source folder does not exist.")
            self.log_message(f"Error: Source folder does not exist: {source_path}", error=True)
            return
            
        if not output_path:
            QMessageBox.warning(self, "Warning", "Please select an output folder.")
            self.log_message("Error: No output folder selected", error=True)
            return
            
        if not os.path.exists(output_path):
            response = QMessageBox.question(
                self, 
                "Create Folder", 
                f"Output folder does not exist. Create it?",
                QMessageBox.Yes | QMessageBox.No
            )
            if response == QMessageBox.Yes:
                try:
                    os.makedirs(output_path)
                    self.log_message(f"Created output folder: {output_path}")
                except Exception as e:
                    error_msg = f"Could not create folder: {str(e)}"
                    QMessageBox.critical(self, "Error", error_msg)
                    self.log_message(f"Error: {error_msg}", error=True)
                    return
            else:
                self.log_message("Processing cancelled: Output folder not created", error=True)
                return
        
        # Save settings before processing
        self.save_raw_settings()
        self.log_message("RAW settings saved")
        
        # Check if rawpy is installed
        try:
            import rawpy
        except ImportError:
            QMessageBox.critical(self, "Missing Dependency", 
                                "The rawpy module is not installed. Please install it with 'pip install rawpy'")
            self.log_message("Error: The rawpy module is not installed. Please install it with 'pip install rawpy'", error=True)
            return
        
        # Collect RAW processing settings
        raw_settings = {
            'source_path': source_path,
            'output_path': output_path,
            'white_balance': self.wb_dropdown.currentText().lower(),
            'bit_depth': 16 if self.bit_depth_dropdown.currentText() == "16-bit" else 8,
            'dpi': 300 if self.raw_dpi_300_radio.isChecked() else 600,
            'compression': self.raw_compression_dropdown.currentText().lower(),
            # Add some metadata from the regular settings
            'photographer': self.photographer_input.text(),
            'institution': self.institution_input.text(),
            'copyright_notice': self.copyright_input.text(),
            'usage_terms': self.usage_terms_input.text(),
            'credit_line': self.credit_input.toPlainText()
        }
        
        # Log important settings
        self.log_message(f"Source path: {source_path}")
        self.log_message(f"Output path: {output_path}")
        self.log_message(f"White Balance: {raw_settings['white_balance']}")
        self.log_message(f"Bit Depth: {raw_settings['bit_depth']}-bit")
        self.log_message(f"DPI: {raw_settings['dpi']}")
        self.log_message(f"Compression: {raw_settings['compression']}")
        
        # Show progress bar
        self.progress_bar.setVisible(True)
        self.progress_bar.setValue(0)
        
        # Disable interface during processing
        self.setEnabled(False)
        
        # Import and start the RawProcessor
        try:
            from processing.raw_processor import RawProcessor
            
            # Start processing in a separate thread
            self.raw_processor = RawProcessor(raw_settings)
            self.raw_processor.progress_update.connect(self.update_progress)
            self.raw_processor.processing_complete.connect(self.raw_processing_finished)
            self.raw_processor.error_occurred.connect(self.handle_error)
            self.raw_processor.log_message.connect(self.log_message)
            self.raw_processor.start()
            
        except ImportError:
            QMessageBox.critical(self, "Missing Module", 
                                "The RawProcessor module is not available in your installation.")
            self.log_message("Error: The RawProcessor module is not available", error=True)
            self.setEnabled(True)
            self.progress_bar.setVisible(False)

    def raw_processing_finished(self, success):
        """Handle completion of RAW processing"""
        self.setEnabled(True)
        self.progress_bar.setVisible(False)
        
        if success:
            QMessageBox.information(self, "Complete", "RAW file processing completed successfully!")
        else:
            QMessageBox.warning(self, "Warning", "RAW processing completed with warnings or errors. Check the log for details.")
                    
    def create_bg_removal_tab(self):
        """Create Background Removal tab with preview functionality and parameter controls"""
        tab = QWidget()
        layout = QVBoxLayout(tab)
        
        # Top section: Source/Destination + Preview
        top_layout = QHBoxLayout()
        
        # Left side: Folders Panel
        folders_group = QGroupBox("Source and Destination")
        folders_layout = QVBoxLayout(folders_group)
        
        # Source Folder
        source_layout = QHBoxLayout()
        source_layout.addWidget(QLabel("Source Folder:"))
        self.bg_source_input = QLineEdit()
        source_layout.addWidget(self.bg_source_input)
        self.bg_source_browse = QPushButton("Browse")
        self.bg_source_browse.clicked.connect(self.browse_bg_source)
        source_layout.addWidget(self.bg_source_browse)
        folders_layout.addLayout(source_layout)
        
        # Output Folder
        output_layout = QHBoxLayout()
        output_layout.addWidget(QLabel("Output Folder:"))
        self.bg_output_input = QLineEdit()
        output_layout.addWidget(self.bg_output_input)
        self.bg_output_browse = QPushButton("Browse")
        self.bg_output_browse.clicked.connect(self.browse_bg_output)
        output_layout.addWidget(self.bg_output_browse)
        folders_layout.addLayout(output_layout)
        
        # Select sample button
        select_sample_layout = QHBoxLayout()
        self.select_sample_button = QPushButton("Select Sample Image for Preview")
        self.select_sample_button.clicked.connect(self.select_sample_image)
        select_sample_layout.addWidget(self.select_sample_button)
        folders_layout.addLayout(select_sample_layout)
        
        top_layout.addWidget(folders_group)
        
        # Right side: Preview Panel
        preview_group = QGroupBox("Preview")
        preview_layout = QVBoxLayout(preview_group)
        
        # Preview images
        preview_images_layout = QHBoxLayout()
        
        # Original image
        original_layout = QVBoxLayout()
        original_layout.addWidget(QLabel("Original:"))
        self.original_preview = QLabel()
        self.original_preview.setFixedSize(300, 300)
        self.original_preview.setAlignment(Qt.AlignCenter)
        self.original_preview.setStyleSheet("border: 1px solid #cccccc")
        original_layout.addWidget(self.original_preview)
        preview_images_layout.addLayout(original_layout)
        
        # Processed image
        processed_layout = QVBoxLayout()
        processed_layout.addWidget(QLabel("Processed:"))
        self.processed_preview = QLabel()
        self.processed_preview.setFixedSize(300, 300)
        self.processed_preview.setAlignment(Qt.AlignCenter)
        self.processed_preview.setStyleSheet("border: 1px solid #cccccc")
        processed_layout.addWidget(self.processed_preview)
        preview_images_layout.addLayout(processed_layout)
        
        preview_layout.addLayout(preview_images_layout)
        
        # Add "Update Preview" button
        self.update_preview_button = QPushButton("Update Preview")
        self.update_preview_button.clicked.connect(self.update_bg_preview)
        preview_layout.addWidget(self.update_preview_button)
        
        top_layout.addWidget(preview_group)
        layout.addLayout(top_layout)
        
        # Background Removal Options
        bg_options_group = QGroupBox("Background Removal Options")
        bg_options_layout = QVBoxLayout(bg_options_group)
        
        # Method Selection
        method_layout = QHBoxLayout()
        method_layout.addWidget(QLabel("Method:"))
        self.bg_method_dropdown = QComboBox()
        self.bg_method_dropdown.addItems([
            "Method 1 (Gaussian + Otsu)", 
            "Method 2 (Simple Thresholding)", 
            "Method 3 (HSV Space)",
            "Method 4 (Black Background)",
            "Method 5 (Edge Detection)",
            "Method 8 (Neural Network)",
            "Method 10 (U-Net or DeepLabV3)",
            "Method 11 (Detectron2 - Photoshop-like)",
        ])
        self.bg_method_dropdown.setCurrentIndex(1)  # Default to Method 2
        method_layout.addWidget(self.bg_method_dropdown)
        bg_options_layout.addLayout(method_layout)
        
        # Method Description
        self.bg_method_desc = QLabel("Method 2: Uses simple thresholding for best quality while maintaining good performance.")
        self.bg_method_desc.setWordWrap(True)
        bg_options_layout.addWidget(self.bg_method_desc)
        
        # Connect method dropdown to description updater
        self.bg_method_dropdown.currentIndexChanged.connect(self.update_bg_method_description)
        
        
        
        layout.addWidget(bg_options_group)
        
        # Parameter Control Panel
        self.param_group = QGroupBox("Method Parameters")
        self.param_layout = QVBoxLayout(self.param_group)
        
        # This will hold our parameter widgets
        self.param_widgets = {}
        
        # Create initial parameter widgets (empty at first)
        self.create_parameter_widgets()
        
        layout.addWidget(self.param_group)
        
        # Processing Button
        button_layout = QHBoxLayout()
        self.process_bg_button = QPushButton("Remove Backgrounds")
        self.process_bg_button.clicked.connect(self.process_backgrounds)
        button_layout.addWidget(self.process_bg_button)
        
        layout.addLayout(button_layout)
        
        # Add stretch at the bottom
        layout.addStretch()
        
        # Add the tab
        self.tabs.addTab(tab, "Background Removal")
        
        # Load saved settings for Background Removal tab
        self.load_bg_settings()
        
        # Connect method dropdown to parameter panel updater
        self.bg_method_dropdown.currentIndexChanged.connect(self.update_parameter_widgets)
        
        # Store the sample image path
        self.sample_image_path = None
        
        # Connect signals for auto-preview updates
        self.bg_method_dropdown.currentIndexChanged.connect(self.queue_preview_update)
        
    def select_sample_image(self):
        """Select a sample image for preview"""
        file, _ = QFileDialog.getOpenFileName(
            self, 
            "Select Sample Image", 
            self.bg_source_input.text(),
            "Image Files (*.png *.jpg *.jpeg *.tif *.tiff)"
        )
        if file:
            self.sample_image_path = file
            self.log_message(f"Selected sample image: {file}")
            self.update_bg_preview()

    def queue_preview_update(self):
        """Queue an update to the preview after a short delay"""
        # Use QTimer to prevent multiple rapid updates when adjusting sliders
        if hasattr(self, 'preview_timer'):
            self.preview_timer.stop()
        else:
            self.preview_timer = QTimer()
            self.preview_timer.setSingleShot(True)
            self.preview_timer.timeout.connect(self.update_bg_preview)
        
        self.preview_timer.start(300)  # Update after 300ms of no changes

    def update_bg_preview(self):
        """Update the background removal preview with current parameters"""
        if not self.sample_image_path or not os.path.exists(self.sample_image_path):
            return
        
        try:
            # Load and display original image
            original_pixmap = QPixmap(self.sample_image_path)
            if not original_pixmap.isNull():
                original_pixmap = original_pixmap.scaled(
                    self.original_preview.width(), 
                    self.original_preview.height(),
                    Qt.KeepAspectRatio,
                    Qt.SmoothTransformation
                )
                self.original_preview.setPixmap(original_pixmap)
                
                # Get current method (1-based)
                method = self.bg_method_dropdown.currentIndex() + 1
                
                # Collect parameters
                params = {}
                if method == 1:  # Method 1 (Gaussian + Otsu)
                    params['clahe_clip'] = self.param_widgets.get('clahe_clip', QDoubleSpinBox()).value()
                    params['block_size'] = self.param_widgets.get('block_size', QSpinBox()).value()
                    params['c_constant'] = self.param_widgets.get('c_constant', QSpinBox()).value()
                    params['dilate_iterations'] = self.param_widgets.get('dilate_iterations', QSpinBox()).value()
                
                elif method == 3:  # HSV Space
                    params['s_threshold'] = self.param_widgets.get('s_threshold', QSpinBox()).value()
                    params['v_threshold'] = self.param_widgets.get('v_threshold', QSpinBox()).value()
                    params['morph_open_size'] = self.param_widgets.get('morph_open_size', QSpinBox()).value()
                    params['morph_close_size'] = self.param_widgets.get('morph_close_size', QSpinBox()).value()
                    params['min_contour_area'] = self.param_widgets.get('min_contour_area', QDoubleSpinBox()).value()
                    params['debug'] = False  # Don't debug for preview
                
                elif method == 4:  # Black Background
                    params['block_size'] = self.param_widgets.get('block_size', QSpinBox()).value()
                    params['c_constant'] = self.param_widgets.get('c_constant', QSpinBox()).value()
                
                elif method == 5:  # Edge Detection
                    params['low_threshold'] = self.param_widgets.get('low_threshold', QSpinBox()).value()
                    params['high_threshold'] = self.param_widgets.get('high_threshold', QSpinBox()).value()
                    params['dilation_iterations'] = self.param_widgets.get('dilation_iterations', QSpinBox()).value()
                
                elif method == 11:  # Detectron2
                    params['detectron_confidence'] = self.param_widgets.get('detectron_confidence', QDoubleSpinBox()).value()
                
                
                # Create temporary file for processed image
                with tempfile.NamedTemporaryFile(suffix='.png', delete=False) as temp_file:
                    temp_output = temp_file.name
                
                # Process the image
                try:
                    import cv2
                    from processing.background_remover import BackgroundRemover
                    
                    image = cv2.imread(self.sample_image_path)
                    if image is None:
                        raise ValueError("Could not load image with OpenCV")
                    
                    # Initialize the BackgroundRemover with current settings
                    bg_remover = BackgroundRemover({
                        'bg_remove_method': method,
                        'params': params
                    })
                    
                    # Call the remove_background method directly
                    result = bg_remover.remove_background(self.sample_image_path, temp_output)
                    
                    if result:
                        # Display the processed image
                        processed_pixmap = QPixmap(temp_output)
                        processed_pixmap = processed_pixmap.scaled(
                            self.processed_preview.width(), 
                            self.processed_preview.height(),
                            Qt.KeepAspectRatio,
                            Qt.SmoothTransformation
                        )
                        self.processed_preview.setPixmap(processed_pixmap)
                        
                        # Clean up temp file after a delay
                        QTimer.singleShot(1000, lambda: os.unlink(temp_output))
                    else:
                        self.log_message("Error generating preview: background removal failed", error=True)
                    
                except Exception as e:
                    self.log_message(f"Error generating preview: {str(e)}", error=True)
                    self.log_message(traceback.format_exc(), error=True)
        
        except Exception as e:
            self.log_message(f"Error updating preview: {str(e)}", error=True)
            self.log_message(traceback.format_exc(), error=True)
            
    def create_parameter_widgets(self):
        """Create parameter widgets without overlapping labels"""
        # Clear existing widgets and layout
        while self.param_layout.count():
            item = self.param_layout.takeAt(0)
            if item.widget():
                item.widget().deleteLater()
            elif item.layout():
                self.clear_layout(item.layout())
        
        # Clear parameter widgets dictionary
        self.param_widgets = {}
        
        # Create a new form layout
        form_layout = QFormLayout()
        form_layout.setFormAlignment(Qt.AlignLeft | Qt.AlignTop)
        form_layout.setLabelAlignment(Qt.AlignLeft)
        
        method_index = self.bg_method_dropdown.currentIndex() + 1  # 1-based
        
        if method_index == 1:  # Method 1 (Gaussian + Otsu)
            clahe_spin = QDoubleSpinBox()
            clahe_spin.setRange(1.0, 8.0)
            clahe_spin.setSingleStep(0.5)
            clahe_spin.setValue(4.0)
            form_layout.addRow("CLAHE Clip Limit:", clahe_spin)
            self.param_widgets['clahe_clip'] = clahe_spin
            
            block_spin = QSpinBox()
            block_spin.setRange(3, 51)
            block_spin.setSingleStep(2)
            block_spin.setValue(15)
            form_layout.addRow("Block Size:", block_spin)
            self.param_widgets['block_size'] = block_spin
            
            c_spin = QSpinBox()
            c_spin.setRange(1, 10)
            c_spin.setValue(2)
            form_layout.addRow("C Constant:", c_spin)
            self.param_widgets['c_constant'] = c_spin
            
            dilate_spin = QSpinBox()
            dilate_spin.setRange(0, 5)
            dilate_spin.setValue(1)
            form_layout.addRow("Dilation Iterations:", dilate_spin)
            self.param_widgets['dilate_iterations'] = dilate_spin
            
        elif method_index == 2:  # Method 2 (Simple Thresholding)
            block_spin = QSpinBox()
            block_spin.setRange(3, 51)
            block_spin.setSingleStep(2)
            block_spin.setValue(15)
            form_layout.addRow("Block Size:", block_spin)
            self.param_widgets['block_size'] = block_spin
            
            c_spin = QSpinBox()
            c_spin.setRange(0, 10)
            c_spin.setValue(2)
            form_layout.addRow("C Constant:", c_spin)
            self.param_widgets['c_constant'] = c_spin
            
            pre_blur_spin = QSpinBox()
            pre_blur_spin.setRange(0, 15)
            pre_blur_spin.setValue(5)
            form_layout.addRow("Pre-blur:", pre_blur_spin)
            self.param_widgets['pre_blur'] = pre_blur_spin
            
            post_blur_spin = QSpinBox()
            post_blur_spin.setRange(0, 15)
            post_blur_spin.setValue(0)
            form_layout.addRow("Post-blur:", post_blur_spin)
            self.param_widgets['post_blur'] = post_blur_spin
            
        elif method_index == 3:  # Method 3 (HSV Space)
            # Saturation threshold
            s_spin = QSpinBox()
            s_spin.setRange(0, 255)
            s_spin.setValue(30)
            form_layout.addRow("Saturation Threshold:", s_spin)
            self.param_widgets['s_threshold'] = s_spin
            
            # Value threshold
            v_spin = QSpinBox()
            v_spin.setRange(0, 255)
            v_spin.setValue(30)
            form_layout.addRow("Value Threshold:", v_spin)
            self.param_widgets['v_threshold'] = v_spin
            
            # Morph open size
            open_spin = QSpinBox()
            open_spin.setRange(1, 15)
            open_spin.setValue(3)
            form_layout.addRow("Morph Open Size:", open_spin)
            self.param_widgets['morph_open_size'] = open_spin
            
            # Morph close size
            close_spin = QSpinBox()
            close_spin.setRange(1, 15)
            close_spin.setValue(5)
            form_layout.addRow("Morph Close Size:", close_spin)
            self.param_widgets['morph_close_size'] = close_spin
            
            # Minimum contour area
            area_spin = QDoubleSpinBox()
            area_spin.setRange(0.01, 1.0)
            area_spin.setSingleStep(0.05)
            area_spin.setValue(0.1)
            form_layout.addRow("Min Contour Area (%):", area_spin)
            self.param_widgets['min_contour_area'] = area_spin
            
            # Stray pixel threshold
            stray_spin = QDoubleSpinBox()
            stray_spin.setRange(0.001, 0.5)  # 0.1% to 50% of image area
            stray_spin.setSingleStep(0.005)
            stray_spin.setValue(0.01)  # Default: 1% of image area
            form_layout.addRow("Max Stray Pixel Area (%):", stray_spin)
            self.param_widgets['max_stray_area'] = stray_spin
            
            # Feathering type
            feather_combo = QComboBox()
            feather_combo.addItems(["None", "Gaussian", "Bilateral", "Morphological"])
            feather_combo.setCurrentIndex(1)  # Default to Gaussian
            form_layout.addRow("Feathering Type:", feather_combo)
            self.param_widgets['feather_type'] = feather_combo
            
            # Feather amount
            feather_slider = QSlider(Qt.Horizontal)
            feather_slider.setRange(0, 20)
            feather_slider.setValue(5)
            feather_slider.setTickPosition(QSlider.TicksBelow)
            feather_slider.setTickInterval(1)
            form_layout.addRow("Feathering Amount:", feather_slider)
            self.param_widgets['feather_amount'] = feather_slider
            
            # Feather value display
            feather_value = QLabel("5")
            feather_value.setAlignment(Qt.AlignCenter)
            form_layout.addRow("", feather_value)  # Empty label for alignment
            
            # Connect slider to value display
            feather_slider.valueChanged.connect(lambda v: feather_value.setText(str(v)))
            
            # Debug checkbox
            debug_check = QCheckBox()
            debug_check.setChecked(False)
            form_layout.addRow("Debug Output:", debug_check)
            self.param_widgets['debug'] = debug_check
            
        elif method_index == 4:  # Black Background
            block_spin = QSpinBox()
            block_spin.setRange(3, 51)
            block_spin.setSingleStep(2)
            block_spin.setValue(11)
            form_layout.addRow("Block Size:", block_spin)
            self.param_widgets['block_size'] = block_spin
            
            c_spin = QSpinBox()
            c_spin.setRange(1, 20)
            c_spin.setValue(2)
            form_layout.addRow("C Constant:", c_spin)
            self.param_widgets['c_constant'] = c_spin
        
        elif method_index == 5:  # Edge Detection
            low_spin = QSpinBox()
            low_spin.setRange(0, 255)
            low_spin.setValue(30)
            form_layout.addRow("Low Threshold:", low_spin)
            self.param_widgets['low_threshold'] = low_spin
            
            high_spin = QSpinBox()
            high_spin.setRange(0, 255)
            high_spin.setValue(100)
            form_layout.addRow("High Threshold:", high_spin)
            self.param_widgets['high_threshold'] = high_spin
            
            dilate_spin = QSpinBox()
            dilate_spin.setRange(1, 10)
            dilate_spin.setValue(3)
            form_layout.addRow("Dilation Iterations:", dilate_spin)
            self.param_widgets['dilation_iterations'] = dilate_spin
        
        
            
        elif method_index == 8:  # Neural Network (rembg)
            model_combo = QComboBox()
            model_combo.addItems(["u2net (High Quality)", "u2netp (Fast)"])
            form_layout.addRow("Model:", model_combo)
            self.param_widgets['rembg_model'] = model_combo
            
            alpha_check = QCheckBox()
            alpha_check.setChecked(True)
            form_layout.addRow("Use Alpha Matting:", alpha_check)
            self.param_widgets['alpha_matting'] = alpha_check
            
            fg_thresh = QSpinBox()
            fg_thresh.setRange(1, 255)
            fg_thresh.setValue(240)
            form_layout.addRow("Foreground Threshold:", fg_thresh)
            self.param_widgets['alpha_fg_threshold'] = fg_thresh
            
            bg_thresh = QSpinBox()
            bg_thresh.setRange(1, 255)
            bg_thresh.setValue(10)
            form_layout.addRow("Background Threshold:", bg_thresh)
            self.param_widgets['alpha_bg_threshold'] = bg_thresh
            
            erode_size = QSpinBox()
            erode_size.setRange(1, 30)
            erode_size.setValue(10)
            form_layout.addRow("Erode Size:", erode_size)
            self.param_widgets['alpha_erode_size'] = erode_size
        
        # Add to create_parameter_widgets
        elif method_index == 10:  # DeepLabV3
            confidence_spin = QDoubleSpinBox()
            confidence_spin.setRange(0.1, 1.0)
            confidence_spin.setSingleStep(0.1)
            confidence_spin.setValue(0.5)
            form_layout.addRow("Confidence Threshold:", confidence_spin)
            self.param_widgets['deeplab_confidence'] = confidence_spin
            
            class_combo = QComboBox()
            class_combo.addItems([
                "Multiple classes",
                "Person (15)",
                "Book (73)", 
                "Dining table (62)",
                "Bowl (84)"
            ])
            form_layout.addRow("Object Class:", class_combo)
            self.param_widgets['deeplab_class'] = class_combo
            
            resolution_combo = QComboBox()
            resolution_combo.addItems(["320×320 (Fast)", "520×520 (Balanced)", "860×860 (Detailed)"])
            resolution_combo.setCurrentIndex(1)  # Default to 520×520
            form_layout.addRow("Input Resolution:", resolution_combo)
            self.param_widgets['deeplab_resolution'] = resolution_combo
        
        elif method_index == 11:  # Detectron2
            confidence_spin = QDoubleSpinBox()
            confidence_spin.setRange(0.1, 1.0)
            confidence_spin.setSingleStep(0.1)
            confidence_spin.setValue(0.7)
            form_layout.addRow("Confidence Threshold:", confidence_spin)
            self.param_widgets['detectron_confidence'] = confidence_spin
            confidence_spin.valueChanged.connect(self.queue_preview_update)
        
        
            
        
        else:  # Methods without parameters
            form_layout.addRow(QLabel("No adjustable parameters for this method."))
        
        # Add the form layout to our parameter group
        self.param_layout.addLayout(form_layout)
        
        # Connect controls to preview updates (for all widgets)
        for widget in self.param_widgets.values():
            if isinstance(widget, (QSpinBox, QDoubleSpinBox, QSlider)):
                widget.valueChanged.connect(self.queue_preview_update)
            elif isinstance(widget, QComboBox):
                widget.currentIndexChanged.connect(self.queue_preview_update)
        else:  # Methods without parameters
            form_layout.addRow(QLabel("No adjustable parameters for this method."))
        
        # Add the form layout to our parameter group
        self.param_layout.addLayout(form_layout)
        
        # Connect controls to preview updates
        for widget in self.param_widgets.values():
            if isinstance(widget, (QSpinBox, QDoubleSpinBox, QSlider)):
                widget.valueChanged.connect(self.queue_preview_update)
            elif isinstance(widget, QComboBox):
                widget.currentIndexChanged.connect(self.queue_preview_update)

    def clear_layout(self, layout):
        """Recursively clear a layout and its widgets"""
        while layout.count():
            item = layout.takeAt(0)
            if item.widget():
                item.widget().deleteLater()
            elif item.layout():
                self.clear_layout(item.layout())
    def update_parameter_widgets(self):
        """Update parameter widgets when method changes"""
        self.create_parameter_widgets()
        self.update_bg_method_description()

    def process_backgrounds(self):
        """Start background removal processing"""
        # Clear previous log
        self.clear_log()
        self.log_message("Starting background removal...")
        
        # Validate source and destination folders
        source_path = self.bg_source_input.text()
        output_path = self.bg_output_input.text()
        
        if not source_path:
            QMessageBox.warning(self, "Warning", "Please select a source folder.")
            self.log_message("Error: No source folder selected", error=True)
            return
            
        if not os.path.exists(source_path):
            QMessageBox.warning(self, "Warning", "Source folder does not exist.")
            self.log_message(f"Error: Source folder does not exist: {source_path}", error=True)
            return
            
        if not output_path:
            QMessageBox.warning(self, "Warning", "Please select an output folder.")
            self.log_message("Error: No output folder selected", error=True)
            return
            
        if not os.path.exists(output_path):
            response = QMessageBox.question(
                self, 
                "Create Folder", 
                f"Output folder does not exist. Create it?",
                QMessageBox.Yes | QMessageBox.No
            )
            if response == QMessageBox.Yes:
                try:
                    os.makedirs(output_path)
                    self.log_message(f"Created output folder: {output_path}")
                except Exception as e:
                    error_msg = f"Could not create folder: {str(e)}"
                    QMessageBox.critical(self, "Error", error_msg)
                    self.log_message(f"Error: {error_msg}", error=True)
                    return
            else:
                self.log_message("Processing cancelled: Output folder not created", error=True)
                return
        
        # Get the selected method
        method = self.bg_method_dropdown.currentIndex() + 1  # 1-based
        
        # Check for Detectron2 dependency if method 11 is selected
        if method == 11:  # Detectron2
            try:
                import detectron2
            except ImportError:
                QMessageBox.critical(
                    self, 
                    "Missing Dependency",
                    "Detectron2 is not installed. Please install with:\n\n"
                    "pip install torch torchvision\n"
                    "pip install detectron2 -f https://dl.fbaipublicfiles.com/detectron2/wheels/cu102/torch1.9/index.html"
                )
                self.log_message("Error: Detectron2 not installed", error=True)
                return
                
        # Save settings before processing
        self.save_bg_settings()
        self.log_message("Background removal settings saved")
        
        # Check if OpenCV is available
        try:
            import cv2
        except ImportError:
            QMessageBox.critical(self, "Missing Dependency", 
                                "OpenCV (cv2) is not installed. Please install it with 'pip install opencv-python'")
            self.log_message("Error: OpenCV (cv2) is not installed. Please install it with 'pip install opencv-python'", error=True)
            return
        
        # Get the selected method
        method = self.bg_method_dropdown.currentIndex() + 1  # 1-based
        
        # Collect parameters based on selected method
        params = {}
        # In process_backgrounds method
        if method == 1:  # Gaussian + Otsu
            params['clahe_clip'] = self.param_widgets.get('clahe_clip', QDoubleSpinBox()).value()
            params['block_size'] = self.param_widgets.get('block_size', QSpinBox()).value()
            params['c_constant'] = self.param_widgets.get('c_constant', QSpinBox()).value()
            params['dilate_iterations'] = self.param_widgets.get('dilate_iterations', QSpinBox()).value()
        # When method 2 is selected
        elif method == 2:  # Method 2 (Simple Thresholding)
            params['block_size'] = self.param_widgets.get('block_size', QSpinBox()).value()
            params['c_constant'] = self.param_widgets.get('c_constant', QSpinBox()).value()
            params['pre_blur'] = self.param_widgets.get('pre_blur', QSpinBox()).value()
            params['post_blur'] = self.param_widgets.get('post_blur', QSpinBox()).value()
            
        elif method == 3:  # HSV Space
            params['s_threshold'] = self.param_widgets.get('s_threshold', QSpinBox()).value()
            params['v_threshold'] = self.param_widgets.get('v_threshold', QSpinBox()).value()
            params['morph_open_size'] = self.param_widgets.get('morph_open_size', QSpinBox()).value()
            params['morph_close_size'] = self.param_widgets.get('morph_close_size', QSpinBox()).value()
            params['min_contour_area'] = self.param_widgets.get('min_contour_area', QDoubleSpinBox()).value()
            params['max_stray_area'] = self.param_widgets.get('max_stray_area', QDoubleSpinBox()).value()
            params['debug'] = self.param_widgets.get('debug', QCheckBox()).isChecked()
            feather_map = {
                0: None, 
                1: 'gaussian', 
                2: 'bilateral', 
                3: 'morph'
            }
            params['feather_type'] = feather_map.get(
                self.param_widgets.get('feather_type', QComboBox()).currentIndex(),
                None
            )
            params['feather_amount'] = self.param_widgets.get('feather_amount', QSlider()).value()
            
        elif method == 4:  # Black Background
            params['block_size'] = self.param_widgets.get('block_size', QSpinBox()).value()
            params['c_constant'] = self.param_widgets.get('c_constant', QSpinBox()).value()
        elif method == 5:  # Edge Detection
            params['low_threshold'] = self.param_widgets.get('low_threshold', QSpinBox()).value()
            params['high_threshold'] = self.param_widgets.get('high_threshold', QSpinBox()).value()
            params['dilation_iterations'] = self.param_widgets.get('dilation_iterations', QSpinBox()).value()
        
        elif method == 8:  # Neural Network (rembg)
            model_map = {0: 'u2net', 1: 'u2netp'}
            model_idx = self.param_widgets.get('rembg_model', QComboBox()).currentIndex()
            params['model_name'] = model_map.get(model_idx, 'u2net')
            
            params['alpha_matting'] = self.param_widgets.get('alpha_matting', QCheckBox()).isChecked()
            params['alpha_matting_foreground_threshold'] = self.param_widgets.get('alpha_fg_threshold', QSpinBox()).value()
            params['alpha_matting_background_threshold'] = self.param_widgets.get('alpha_bg_threshold', QSpinBox()).value()
            params['alpha_matting_erode_size'] = self.param_widgets.get('alpha_erode_size', QSpinBox()).value()
        elif method == 11:  # Detectron2
            params['detectron_confidence'] = self.param_widgets.get('detectron_confidence', QDoubleSpinBox()).value()
        # Add this to the method selection (around line 1560)
       
        
        # Collect background removal settings
        bg_settings = {
            'source_path': source_path,
            'output_path': output_path,
            'bg_remove_method': method,
            
            'params': params
        }
        
        # Log important settings
        self.log_message(f"Source path: {source_path}")
        self.log_message(f"Output path: {output_path}")
        self.log_message(f"Method: {bg_settings['bg_remove_method']}")
        
        
        # Show progress bar
        self.progress_bar.setVisible(True)
        self.progress_bar.setValue(0)
        
        # Disable interface during processing
        self.setEnabled(False)
        
        # Import and start the BackgroundRemover
        try:
            from processing.background_remover import BackgroundRemover
            
            # Start processing in a separate thread
            self.bg_remover = BackgroundRemover(bg_settings)
            self.bg_remover.progress_update.connect(self.update_progress)
            self.bg_remover.processing_complete.connect(self.bg_processing_finished)
            self.bg_remover.error_occurred.connect(self.handle_error)
            self.bg_remover.log_message.connect(self.log_message)
            self.bg_remover.start()
            
        except ImportError:
            QMessageBox.critical(self, "Missing Module", 
                                "The BackgroundRemover module is not available in your installation.")
            self.log_message("Error: The BackgroundRemover module is not available", error=True)
            self.setEnabled(True)
            self.progress_bar.setVisible(False)
        
    def load_bg_settings(self):
        """Load saved settings for Background Removal tab"""
        self.bg_source_input.setText(self.settings.value("bg_source_path", ""))
        self.bg_output_input.setText(self.settings.value("bg_output_path", ""))
        
        # Method
        bg_method = self.settings.value("bg_remove_method", 1, type=int)
        self.bg_method_dropdown.setCurrentIndex(bg_method - 1)  # Convert from 1-based to 0-based index
        
        
        
        # Update method description
        self.update_bg_method_description()

    def save_bg_settings(self):
        """Save Background Removal settings"""
        self.settings.setValue("bg_source_path", self.bg_source_input.text())
        self.settings.setValue("bg_output_path", self.bg_output_input.text())
        self.settings.setValue("bg_remove_method", self.bg_method_dropdown.currentIndex() + 1)  # Store as 1-based
        

    def browse_bg_source(self):
        """Browse for Background Removal source folder"""
        folder = QFileDialog.getExistingDirectory(self, "Select Source Folder with Images", self.bg_source_input.text())
        if folder:
            self.bg_source_input.setText(folder)

    def browse_bg_output(self):
        """Browse for Background Removal output folder"""
        folder = QFileDialog.getExistingDirectory(self, "Select Output Folder for Processed Images", self.bg_output_input.text())
        if folder:
            self.bg_output_input.setText(folder)

    def update_bg_method_description(self):
        """Update the description text based on selected method"""
        method_index = self.bg_method_dropdown.currentIndex()
        
        descriptions = [
            "Method 1: Uses Gaussian blur and color binning with Otsu thresholding. Slower but can handle noisy backgrounds.",
            "Method 2: Uses simple thresholding for best quality while maintaining good performance. Works best with white backgrounds.",
            "Method 3: Uses HSV color space with adjustable thresholds for saturation and value. Optimized for cuneiform tablets against dark backgrounds with configurable parameters.",
            "Method 4: Optimized for objects with black backgrounds. Uses adaptive thresholding to handle varying lighting conditions.",
            "Method 5: Uses edge detection to find object boundaries. Works well for objects with clear edges against any background color.",
            "Method 8: Uses neural network segmentation for high-quality automatic background removal. Requires rembg library installation.",
            "Method 10: Uses pre-trained deep learning models (U-Net or DeepLabV3) for advanced segmentation. Requires PyTorch installation.",
            "Method 11: Uses Detectron2's Mask R-CNN for Photoshop-like object selection. Provides high-quality automatic object detection and segmentation. Requires Detectron2 installation.",
           
        ]
        
        
        if 0 <= method_index < len(descriptions):
            self.bg_method_desc.setText(descriptions[method_index])

    def bg_processing_finished(self, success):
        """Handle completion of background removal processing"""
        self.setEnabled(True)
        self.progress_bar.setVisible(False)
        
        if success:
            QMessageBox.information(self, "Complete", "Background removal completed successfully!")
        else:
            QMessageBox.warning(self, "Warning", "Background removal completed with warnings or errors. Check the log for details.")
            