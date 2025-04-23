#!/usr/bin/env python3
"""
Main window implementation for the Cuneiform Tablet Processor application with icon-only vertical toolbar using Qt standard icons.
"""

import os
import sys
import traceback
import cv2
import numpy as np
import torch
import tempfile
from PyQt5.QtWidgets import (QMainWindow, QWidget, QVBoxLayout, QHBoxLayout, 
                            QLabel, QLineEdit, QPushButton, QFileDialog, QCheckBox, 
                            QRadioButton, QButtonGroup, QSpinBox, QDoubleSpinBox, 
                            QComboBox, QProgressBar, QMessageBox, QGroupBox, QSlider, 
                            QTextEdit, QPlainTextEdit, QDockWidget, QFormLayout, QScrollArea,
                            QGraphicsView, QGraphicsScene, QGraphicsPixmapItem, QToolBar,
                            QAction, QStackedWidget, QStyle)
from PyQt5.QtCore import Qt, QSettings, pyqtSignal, QTimer, QPointF
from PyQt5.QtGui import QPixmap, QIcon, QTextCursor, QColor, QTransform
from PyQt5.QtGui import QPainter

# Try to import RawProcessor and BackgroundRemover
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

class ZoomableGraphicsView(QGraphicsView):
    """Custom graphics view with zoom functionality"""
    def __init__(self, scene):
        super().__init__(scene)
        self.setRenderHint(QPainter.Antialiasing)
        self.setRenderHint(QPainter.SmoothPixmapTransform)
        self.setDragMode(QGraphicsView.ScrollHandDrag)
        self.setTransformationAnchor(QGraphicsView.AnchorUnderMouse)
        self._zoom = 1.0
        self.setMinimumSize(400, 400)

    def wheelEvent(self, event):
        """Handle mouse wheel for zooming"""
        zoom_in_factor = 1.25
        zoom_out_factor = 1 / zoom_in_factor

        if event.angleDelta().y() > 0:
            zoom_factor = zoom_in_factor
        else:
            zoom_factor = zoom_out_factor

        self._zoom *= zoom_factor
        self._zoom = max(0.1, min(self._zoom, 5.0))

        self.resetTransform()
        self.scale(self._zoom, self._zoom)

class MainWindow(QMainWindow):
    """Main application window with left (toolbar and controls) and right (preview/log) sections"""
    def __init__(self):
        super().__init__()
        self.setWindowTitle("Cuneiform Tablet Processor")
        self.setMinimumSize(1000, 700)
        
        # Load settings
        self.settings = QSettings("TabletProcessor", "CuneiformApp")
        
        # Create main widget and layout
        self.central_widget = QWidget()
        self.setCentralWidget(self.central_widget)
        self.main_layout = QHBoxLayout(self.central_widget)
        
        # Create left section (controls)
        self.controls_widget = QWidget()
        self.controls_layout = QVBoxLayout(self.controls_widget)
        self.controls_widget.setMaximumWidth(400)
        
        # Create right section (preview and log)
        self.preview_widget = QWidget()
        self.preview_layout = QVBoxLayout(self.preview_widget)
        
        # Split layout: ~40% controls, ~60% preview/log
        self.main_layout.addWidget(self.controls_widget, 4)
        self.main_layout.addWidget(self.preview_widget, 5)
        
        # Create toolbar and content area
        self.controls_inner_layout = QHBoxLayout()
        self.toolbar = QToolBar()
        self.toolbar.setOrientation(Qt.Vertical)
        self.toolbar.setFixedWidth(50)  # Compact toolbar
        self.toolbar.setStyleSheet("""
            QToolBar { border: none; spacing: 2px; }
            QToolButton { border: none; padding: 5px; }
        """)
        
        self.content_stack = QStackedWidget()
        self.content_actions = {}  # Map action to content name
        self.content_indices = {}  # Map content name to stack index
        
        self.controls_inner_layout.addWidget(self.toolbar)
        self.controls_inner_layout.addWidget(self.content_stack)
        self.controls_layout.addLayout(self.controls_inner_layout)
        
        # Create toolbar actions and content
        self.create_toolbar()
        
        # Create preview area
        self.create_preview_area()
        
        # Create buttons (progress bar)
        self.create_buttons()
        
        # Create log area
        self.create_log_area()
        
        # Load saved settings
        self.load_settings()
        
        # Initialize processors
        self.processor = None
        self.raw_processor = None
        self.bg_remover = None
    
    def create_toolbar(self):
        """Create toolbar actions with Qt standard icons and their content"""
        # Processing Options
        proc_action = QAction(self.style().standardIcon(QStyle.SP_BrowserReload), "", self)
        proc_action.setToolTip("Processing Options")
        proc_action.setCheckable(True)
        proc_action.setChecked(True)  # Default selection
        proc_widget = self.create_processing_tab()
        self.content_actions[proc_action] = "Processing Options"
        self.content_indices["Processing Options"] = self.content_stack.addWidget(proc_widget)
        self.toolbar.addAction(proc_action)
        
        # Metadata
        meta_action = QAction(self.style().standardIcon(QStyle.SP_FileIcon), "", self)
        meta_action.setToolTip("Metadata")
        meta_action.setCheckable(True)
        meta_widget = self.create_metadata_tab()
        self.content_actions[meta_action] = "Metadata"
        self.content_indices["Metadata"] = self.content_stack.addWidget(meta_widget)
        self.toolbar.addAction(meta_action)
        
        # RAW Processing (if available)
        if RAW_PROCESSING_AVAILABLE:
            raw_action = QAction(self.style().standardIcon(QStyle.SP_DriveFDIcon), "", self)
            raw_action.setToolTip("RAW Processing")
            raw_action.setCheckable(True)
            raw_widget = self.create_raw_processing_tab()
            self.content_actions[raw_action] = "RAW Processing"
            self.content_indices["RAW Processing"] = self.content_stack.addWidget(raw_widget)
            self.toolbar.addAction(raw_action)
        
        # Background Removal (if available)
        if BACKGROUND_REMOVAL_AVAILABLE:
            bg_action = QAction(self.style().standardIcon(QStyle.SP_TrashIcon), "", self)
            bg_action.setToolTip("Background Removal")
            bg_action.setCheckable(True)
            bg_widget = self.create_bg_removal_tab()
            self.content_actions[bg_action] = "Background Removal"
            self.content_indices["Background Removal"] = self.content_stack.addWidget(bg_widget)
            self.toolbar.addAction(bg_action)
        
        # Connect actions and ensure single selection
        for action in self.content_actions:
            action.triggered.connect(self.switch_content)
        
        # Set initial content
        self.content_stack.setCurrentIndex(self.content_indices["Processing Options"])
    
    def switch_content(self):
        """Switch content when a toolbar action is triggered"""
        action = self.sender()
        if action and action.isChecked():
            # Uncheck other actions
            for other_action in self.content_actions:
                if other_action != action:
                    other_action.setChecked(False)
            # Switch content
            content_name = self.content_actions[action]
            self.content_stack.setCurrentIndex(self.content_indices[content_name])
        elif not any(a.isChecked() for a in self.content_actions):
            # If no action is checked, restore the sender's check state
            action.setChecked(True)
    
    def create_preview_area(self):
        """Create the preview area with zoom functionality"""
        preview_group = QGroupBox("Image Preview")
        preview_layout = QVBoxLayout(preview_group)
        
        self.graphics_scene = QGraphicsScene()
        self.graphics_view = ZoomableGraphicsView(self.graphics_scene)
        self.preview_item = QGraphicsPixmapItem()
        self.graphics_scene.addItem(self.preview_item)
        
        zoom_controls = QHBoxLayout()
        zoom_in_button = QPushButton("+")
        zoom_in_button.clicked.connect(self.zoom_in)
        zoom_out_button = QPushButton("-")
        zoom_out_button.clicked.connect(self.zoom_out)
        reset_zoom_button = QPushButton("Reset Zoom")
        reset_zoom_button.clicked.connect(self.reset_zoom)
        
        zoom_controls.addWidget(QLabel("Zoom:"))
        zoom_controls.addWidget(zoom_in_button)
        zoom_controls.addWidget(zoom_out_button)
        zoom_controls.addWidget(reset_zoom_button)
        zoom_controls.addStretch()
        
        preview_layout.addWidget(self.graphics_view)
        preview_layout.addLayout(zoom_controls)
        
        self.preview_layout.addWidget(preview_group, 2)  # Preview takes more space
    
    def create_log_area(self):
        """Create the log area below the preview in the right section"""
        log_group = QGroupBox("Processing Log")
        log_layout = QVBoxLayout(log_group)
        
        self.log_text = QPlainTextEdit()
        self.log_text.setReadOnly(True)
        self.log_text.setMaximumBlockCount(1000)
        self.log_text.setMaximumHeight(200)  # Limit log height
        
        log_controls = QHBoxLayout()
        clear_button = QPushButton("Clear Log")
        clear_button.clicked.connect(self.clear_log)
        log_controls.addStretch()
        log_controls.addWidget(clear_button)
        
        log_layout.addWidget(self.log_text)
        log_layout.addLayout(log_controls)
        
        self.preview_layout.addWidget(log_group, 1)  # Log takes less space
    
    def create_buttons(self):
        """Create the progress bar"""
        self.progress_bar = QProgressBar()
        self.progress_bar.setVisible(False)
        self.controls_layout.addWidget(self.progress_bar)

    def create_processing_tab(self):
        """Create Processing Options content"""
        tab = QWidget()
        scroll = QScrollArea()
        scroll.setWidget(tab)
        scroll.setWidgetResizable(True)
        layout = QVBoxLayout(tab)
        
        folders_group = QGroupBox("Source and Destination")
        folders_layout = QVBoxLayout(folders_group)
        
        source_layout = QHBoxLayout()
        source_layout.addWidget(QLabel("Source Folder:"))
        self.source_input = QLineEdit()
        source_layout.addWidget(self.source_input)
        self.source_browse = QPushButton("Browse")
        self.source_browse.clicked.connect(self.browse_source)
        source_layout.addWidget(self.source_browse)
        folders_layout.addLayout(source_layout)
        
        output_layout = QHBoxLayout()
        output_layout.addWidget(QLabel("Output Folder:"))
        self.output_input = QLineEdit()
        output_layout.addWidget(self.output_input)
        self.output_browse = QPushButton("Browse")
        self.output_browse.clicked.connect(self.browse_output)
        output_layout.addWidget(self.output_browse)
        folders_layout.addLayout(output_layout)
        
        layout.addWidget(folders_group)
        
        logo_group = QGroupBox("Logo Options")
        logo_layout = QVBoxLayout(logo_group)
        
        logo_check_layout = QHBoxLayout()
        self.add_logo_checkbox = QCheckBox("Add Logo to Images")
        self.add_logo_checkbox.stateChanged.connect(self.toggle_logo_controls)
        logo_check_layout.addWidget(self.add_logo_checkbox)
        logo_layout.addLayout(logo_check_layout)
        
        logo_path_layout = QHBoxLayout()
        logo_path_layout.addWidget(QLabel("Logo File:"))
        self.logo_path_input = QLineEdit()
        logo_path_layout.addWidget(self.logo_path_input)
        self.logo_browse = QPushButton("Browse")
        self.logo_browse.clicked.connect(self.browse_logo)
        logo_path_layout.addWidget(self.logo_browse)
        logo_layout.addLayout(logo_path_layout)
        
        layout.addWidget(logo_group)
        
        tiff_group = QGroupBox("TIFF Options")
        tiff_layout = QVBoxLayout(tiff_group)
        
        dpi_layout = QHBoxLayout()
        dpi_layout.addWidget(QLabel("Final Resolution:"))
        dpi_radio_group = QHBoxLayout()
        self.dpi_300_radio = QRadioButton("300 DPI")
        self.dpi_600_radio = QRadioButton("600 DPI")
        dpi_radio_group.addWidget(self.dpi_300_radio)
        dpi_radio_group.addWidget(self.dpi_600_radio)
        dpi_layout.addLayout(dpi_radio_group)
        tiff_layout.addLayout(dpi_layout)
        
        compression_layout = QHBoxLayout()
        compression_layout.addWidget(QLabel("Compression:"))
        self.compression_dropdown = QComboBox()
        self.compression_dropdown.addItems(["None", "LZW", "ZIP"])
        compression_layout.addWidget(self.compression_dropdown)
        tiff_layout.addLayout(compression_layout)
        
        layout.addWidget(tiff_group)
        
        jpeg_group = QGroupBox("JPEG Options")
        jpeg_layout = QVBoxLayout(jpeg_group)
        
        jpeg_check_layout = QHBoxLayout()
        self.save_jpeg_checkbox = QCheckBox("Save JPEG Copy")
        self.save_jpeg_checkbox.stateChanged.connect(self.toggle_jpeg_controls)
        jpeg_check_layout.addWidget(self.save_jpeg_checkbox)
        jpeg_layout.addLayout(jpeg_check_layout)
        
        quality_header_layout = QHBoxLayout()
        quality_header_layout.addWidget(QLabel("JPEG Quality:"))
        self.quality_level_text = QLabel("")
        self.quality_level_text.setMinimumWidth(100)
        quality_header_layout.addWidget(self.quality_level_text)
        jpeg_layout.addLayout(quality_header_layout)
        
        quality_control_layout = QHBoxLayout()
        self.quality_dropdown = QComboBox()
        self.quality_dropdown.addItems([str(i) for i in range(1, 13)])
        self.quality_dropdown.setCurrentIndex(7)
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
        
        self.update_quality_level_text(8)
        
        button_layout = QHBoxLayout()
        self.process_button = QPushButton("Process Images")
        self.process_button.clicked.connect(self.process_images)
        self.cancel_button = QPushButton("Cancel")
        self.cancel_button.clicked.connect(self.close)
        button_layout.addWidget(self.process_button)
        button_layout.addWidget(self.cancel_button)
        layout.addLayout(button_layout)
        
        layout.addStretch()
        return scroll

    def zoom_in(self):
        """Zoom in the preview image"""
        self.graphics_view._zoom *= 1.25
        self.graphics_view._zoom = min(self.graphics_view._zoom, 5.0)
        self.graphics_view.resetTransform()
        self.graphics_view.scale(self._zoom, self._zoom)
    
    def zoom_out(self):
        """Zoom out the preview image"""
        self.graphics_view._zoom /= 1.25
        self.graphics_view._zoom = max(self.graphics_view._zoom, 0.1)
        self.graphics_view.resetTransform()
        self.graphics_view.scale(self._zoom, self._zoom)
    
    def reset_zoom(self):
        """Reset zoom to 1:1"""
        self.graphics_view._zoom = 1.0
        self.graphics_view.resetTransform()
        self.graphics_view.scale(1.0, 1.0)
    
    def update_preview_image(self):
        """Update the preview image when a sample is selected"""
        if hasattr(self, 'sample_image_path') and self.sample_image_path and os.path.exists(self.sample_image_path):
            pixmap = QPixmap(self.sample_image_path)
            if not pixmap.isNull():
                self.preview_item.setPixmap(pixmap)
                self.graphics_scene.setSceneRect(self.preview_item.boundingRect())
                self.reset_zoom()

    def create_metadata_tab(self):
        """Create Metadata content"""
        tab = QWidget()
        scroll = QScrollArea()
        scroll.setWidget(tab)
        scroll.setWidgetResizable(True)
        layout = QVBoxLayout(tab)
        
        general_group = QGroupBox("General Metadata")
        general_layout = QVBoxLayout(general_group)
        
        photographer_layout = QHBoxLayout()
        photographer_layout.addWidget(QLabel("Photographer:"))
        self.photographer_input = QLineEdit()
        photographer_layout.addWidget(self.photographer_input)
        general_layout.addLayout(photographer_layout)
        
        institution_layout = QHBoxLayout()
        institution_layout.addWidget(QLabel("Institution:"))
        self.institution_input = QLineEdit()
        institution_layout.addWidget(self.institution_input)
        general_layout.addLayout(institution_layout)
        
        copyright_layout = QHBoxLayout()
        copyright_layout.addWidget(QLabel("Copyright Notice:"))
        self.copyright_input = QLineEdit()
        copyright_layout.addWidget(self.copyright_input)
        general_layout.addLayout(copyright_layout)
        
        usage_layout = QHBoxLayout()
        usage_layout.addWidget(QLabel("Copyright (Usage Terms):"))
        self.usage_terms_input = QLineEdit()
        usage_layout.addWidget(self.usage_terms_input)
        general_layout.addLayout(usage_layout)
        
        layout.addWidget(general_group)
        
        iptc_group = QGroupBox("IPTC Metadata")
        iptc_layout = QVBoxLayout(iptc_group)
        
        title_layout = QHBoxLayout()
        title_layout.addWidget(QLabel("Title:"))
        title_display = QLabel("(Will be set to Tablet Number)")
        title_layout.addWidget(title_display)
        iptc_layout.addLayout(title_layout)
        
        headline_layout = QHBoxLayout()
        headline_layout.addWidget(QLabel("Headline:"))
        headline_display = QLabel("(Will be set to Tablet Number)")
        headline_layout.addWidget(headline_display)
        iptc_layout.addLayout(headline_layout)
        
        author_layout = QHBoxLayout()
        author_layout.addWidget(QLabel("Author/Creator:"))
        self.author_display = QLabel("")
        author_layout.addWidget(self.author_display)
        iptc_layout.addLayout(author_layout)
        
        credit_layout = QHBoxLayout()
        credit_layout.addWidget(QLabel("Credit Line:"))
        credit_layout.addStretch(1)
        iptc_layout.addLayout(credit_layout)
        
        self.credit_input = QTextEdit()
        default_credit = ""
        self.credit_input.setText(default_credit)
        self.credit_input.setMaximumHeight(100)
        iptc_layout.addWidget(self.credit_input)
        
        layout.addWidget(iptc_group)
        layout.addStretch()
        
        self.photographer_input.textChanged.connect(self.update_author_display)
        return scroll
    
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
        
        if RAW_PROCESSING_AVAILABLE and hasattr(self, 'raw_source_input'):
            self.save_raw_settings()
        
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
        else:
            value = self.quality_dropdown.currentIndex() + 1
            self.quality_slider.setValue(value)
        
        self.quality_value.setText(str(value))
        self.update_quality_level_text(value)
    
    def update_author_display(self):
        """Update author display when photographer changes"""
        self.author_display.setText(self.photographer_input.text())
    
    def process_images(self):
        """Start image processing"""
        self.clear_log()
        self.log_message("Starting processing...")
        
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
                
        self.save_settings()
        self.log_message("Settings saved")
        
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
        
        self.log_message(f"Source path: {source_path}")
        self.log_message(f"Output path: {output_path}")
        self.log_message(f"DPI: {settings_dict['dpi']}")
        self.log_message(f"Compression: {settings_dict['compression']}")
        self.log_message(f"Save JPEG: {settings_dict['save_jpeg']}")
        if settings_dict['add_logo']:
            self.log_message(f"Logo: {settings_dict['logo_path']}")
        
        self.progress_bar.setVisible(True)
        self.progress_bar.setValue(0)
        
        self.setEnabled(False)
        
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
    
    def log_message(self, message, error=False):
        """Add a message to the log area with timestamp"""
        from datetime import datetime
        timestamp = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
        self.log_text.moveCursor(QTextCursor.End)
        if error:
            self.log_text.appendHtml(f'<span style="color: red">[{timestamp}] {message}</span>')
        else:
            self.log_text.appendPlainText(f"[{timestamp}] {message}")
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
        """Create RAW Processing content"""
        tab = QWidget()
        scroll = QScrollArea()
        scroll.setWidget(tab)
        scroll.setWidgetResizable(True)
        layout = QVBoxLayout(tab)
        
        folders_group = QGroupBox("Source and Destination")
        folders_layout = QVBoxLayout(folders_group)
        
        source_layout = QHBoxLayout()
        source_layout.addWidget(QLabel("Source Folder:"))
        self.raw_source_input = QLineEdit()
        source_layout.addWidget(self.raw_source_input)
        self.raw_source_browse = QPushButton("Browse")
        self.raw_source_browse.clicked.connect(self.browse_raw_source)
        source_layout.addWidget(self.raw_source_browse)
        folders_layout.addLayout(source_layout)
        
        output_layout = QHBoxLayout()
        output_layout.addWidget(QLabel("Output Folder:"))
        self.raw_output_input = QLineEdit()
        output_layout.addWidget(self.raw_output_input)
        self.raw_output_browse = QPushButton("Browse")
        self.raw_output_browse.clicked.connect(self.browse_raw_output)
        output_layout.addWidget(self.raw_output_browse)
        folders_layout.addLayout(output_layout)
        
        layout.addWidget(folders_group)
        
        raw_options_group = QGroupBox("RAW Processing Options")
        raw_options_layout = QVBoxLayout(raw_options_group)
        
        wb_layout = QHBoxLayout()
        wb_layout.addWidget(QLabel("White Balance:"))
        self.wb_dropdown = QComboBox()
        self.wb_dropdown.addItems(["Camera", "Auto", "None"])
        wb_layout.addWidget(self.wb_dropdown)
        raw_options_layout.addLayout(wb_layout)
        
        bit_depth_layout = QHBoxLayout()
        bit_depth_layout.addWidget(QLabel("Output Bit Depth:"))
        self.bit_depth_dropdown = QComboBox()
        self.bit_depth_dropdown.addItems(["16-bit", "8-bit"])
        bit_depth_layout.addWidget(self.bit_depth_dropdown)
        raw_options_layout.addLayout(bit_depth_layout)
        
        dpi_layout = QHBoxLayout()
        dpi_layout.addWidget(QLabel("Final Resolution:"))
        raw_dpi_radio_group = QButtonGroup(self)
        self.raw_dpi_300_radio = QRadioButton("300 DPI")
        self.raw_dpi_600_radio = QRadioButton("600 DPI")
        self.raw_dpi_600_radio.setChecked(True)
        raw_dpi_radio_group.addButton(self.raw_dpi_300_radio)
        raw_dpi_radio_group.addButton(self.raw_dpi_600_radio)
        dpi_radio_hbox = QHBoxLayout()
        dpi_radio_hbox.addWidget(self.raw_dpi_300_radio)
        dpi_radio_hbox.addWidget(self.raw_dpi_600_radio)
        dpi_layout.addLayout(dpi_radio_hbox)
        raw_options_layout.addLayout(dpi_layout)
        
        compression_layout = QHBoxLayout()
        compression_layout.addWidget(QLabel("Compression:"))
        self.raw_compression_dropdown = QComboBox()
        self.raw_compression_dropdown.addItems(["None", "LZW", "ZIP"])
        compression_layout.addWidget(self.raw_compression_dropdown)
        raw_options_layout.addLayout(compression_layout)
        
        layout.addWidget(raw_options_group)
        
        button_layout = QHBoxLayout()
        self.process_raw_button = QPushButton("Convert RAW Files to TIFF")
        self.process_raw_button.clicked.connect(self.process_raw_files)
        button_layout.addWidget(self.process_raw_button)
        
        layout.addLayout(button_layout)
        layout.addStretch()
        
        self.load_raw_settings()
        return scroll

    def load_raw_settings(self):
        """Load saved settings for RAW processing tab"""
        self.raw_source_input.setText(self.settings.value("raw_source_path", ""))
        self.raw_output_input.setText(self.settings.value("raw_output_path", ""))
        
        wb_value = self.settings.value("raw_white_balance", "Camera")
        index = self.wb_dropdown.findText(wb_value)
        if index >= 0:
            self.wb_dropdown.setCurrentIndex(index)
        
        bit_depth = self.settings.value("raw_bit_depth", "16-bit")
        index = self.bit_depth_dropdown.findText(bit_depth)
        if index >= 0:
            self.bit_depth_dropdown.setCurrentIndex(index)
        
        raw_dpi = self.settings.value("raw_dpi", 600, type=int)
        if raw_dpi == 300:
            self.raw_dpi_300_radio.setChecked(True)
        else:
            self.raw_dpi_600_radio.setChecked(True)
        
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
        self.clear_log()
        self.log_message("Starting RAW processing...")
        
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
        
        self.save_raw_settings()
        self.log_message("RAW settings saved")
        
        try:
            import rawpy
        except ImportError:
            QMessageBox.critical(self, "Missing Dependency", 
                                "The rawpy module is not installed. Please install it with 'pip install rawpy'")
            self.log_message("Error: The rawpy module is not installed. Please install it with 'pip install rawpy'", error=True)
            return
        
        raw_settings = {
            'source_path': source_path,
            'output_path': output_path,
            'white_balance': self.wb_dropdown.currentText().lower(),
            'bit_depth': 16 if self.bit_depth_dropdown.currentText() == "16-bit" else 8,
            'dpi': 300 if self.raw_dpi_300_radio.isChecked() else 600,
            'compression': self.raw_compression_dropdown.currentText().lower(),
            'photographer': self.photographer_input.text(),
            'institution': self.institution_input.text(),
            'copyright_notice': self.copyright_input.text(),
            'usage_terms': self.copyright_input.text(),
            'credit_line': self.credit_input.toPlainText()
        }
        
        self.log_message(f"Source path: {source_path}")
        self.log_message(f"Output path: {output_path}")
        self.log_message(f"White Balance: {raw_settings['white_balance']}")
        self.log_message(f"Bit Depth: {raw_settings['bit_depth']}-bit")
        self.log_message(f"DPI: {raw_settings['dpi']}")
        self.log_message(f"Compression: {raw_settings['compression']}")
        
        self.progress_bar.setVisible(True)
        self.progress_bar.setValue(0)
        
        self.setEnabled(False)
        
        try:
            from processing.raw_processor import RawProcessor
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
        """Create Background Removal content"""
        tab = QWidget()
        scroll = QScrollArea()
        scroll.setWidget(tab)
        scroll.setWidgetResizable(True)
        layout = QVBoxLayout(tab)
        
        folders_group = QGroupBox("Source and Destination")
        folders_layout = QVBoxLayout(folders_group)
        
        source_layout = QHBoxLayout()
        source_layout.addWidget(QLabel("Source Folder:"))
        self.bg_source_input = QLineEdit()
        source_layout.addWidget(self.bg_source_input)
        self.bg_source_browse = QPushButton("Browse")
        self.bg_source_browse.clicked.connect(self.browse_bg_source)
        source_layout.addWidget(self.bg_source_browse)
        folders_layout.addLayout(source_layout)
        
        output_layout = QHBoxLayout()
        output_layout.addWidget(QLabel("Output Folder:"))
        self.bg_output_input = QLineEdit()
        output_layout.addWidget(self.bg_output_input)
        self.bg_output_browse = QPushButton("Browse")
        self.bg_output_browse.clicked.connect(self.browse_bg_output)
        output_layout.addWidget(self.bg_output_browse)
        folders_layout.addLayout(output_layout)
        
        select_sample_layout = QHBoxLayout()
        self.select_sample_button = QPushButton("Select Sample Image for Preview")
        self.select_sample_button.clicked.connect(self.select_sample_image)
        select_sample_layout.addWidget(self.select_sample_button)
        folders_layout.addLayout(select_sample_layout)
        
        layout.addWidget(folders_group)
        
        bg_options_group = QGroupBox("HSV Adaptive Contour Extraction")
        bg_options_layout = QVBoxLayout(bg_options_group)
        
        
        
        self.bg_method_desc = QLabel(
            "Uses advanced HSV color space analysis to intelligently separate cuneiform tablets from backgrounds. This technique employs adjustable saturation and value thresholds combined with morphological operations to create precise masks. Features customizable edge smoothing with multiple feathering options and stray pixel removal for professional results with complex cuneiform tablets."
        )
        self.bg_method_desc.setWordWrap(True)
        bg_options_layout.addWidget(self.bg_method_desc)
        
        layout.addWidget(bg_options_group)
        
        self.param_group = QGroupBox("Method Parameters")
        self.param_layout = QVBoxLayout(self.param_group)
        self.param_widgets = {}
        self.create_parameter_widgets()
        
        layout.addWidget(self.param_group)
        
        button_layout = QHBoxLayout()
        self.process_bg_button = QPushButton("Remove Backgrounds")
        self.process_bg_button.clicked.connect(self.process_backgrounds)
        button_layout.addWidget(self.process_bg_button)
        
        layout.addLayout(button_layout)
        layout.addStretch()
        
        self.load_bg_settings()
        
        self.sample_image_path = None
        return scroll
    
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
            self.update_bg_preview()  # Show processed image

    def queue_preview_update(self):
        """Queue a preview update with a small delay to avoid overloading"""
        QTimer.singleShot(100, self.update_bg_preview)

    def update_bg_preview(self):
        """Update the background removal preview with current parameters"""
        if not self.sample_image_path or not os.path.exists(self.sample_image_path):
            self.log_message("No sample image selected for preview", error=True)
            return

        try:
            params = {
                's_threshold': self.param_widgets.get('s_threshold', QSpinBox()).value(),
                'v_threshold': self.param_widgets.get('v_threshold', QSpinBox()).value(),
                'morph_open_size': self.param_widgets.get('morph_open_size', QSpinBox()).value(),
                'morph_close_size': self.param_widgets.get('morph_close_size', QSpinBox()).value(),
                'min_contour_area': self.param_widgets.get('min_contour_area', QDoubleSpinBox()).value(),
                'max_stray_area': self.param_widgets.get('max_stray_area', QDoubleSpinBox()).value(),
                'feather_type': {0: None, 1: 'gaussian', 2: 'bilateral', 3: 'morph'}.get(
                    self.param_widgets.get('feather_type', QComboBox()).currentIndex(), None
                ),
                'feather_amount': self.param_widgets.get('feather_amount', QSlider()).value(),
                'smoothing_type': {0: None, 1: 'median', 2: 'gaussian'}.get(
                    self.param_widgets.get('smoothing_type', QComboBox()).currentIndex(), None
                ),
                'smoothing_amount': self.param_widgets.get('smoothing_amount', QSlider()).value()
            }

            # Log all parameters
            self.log_message("Updating preview with parameters:")
            self.log_message(f"  Saturation Threshold: {params['s_threshold']}")
            self.log_message(f"  Value Threshold: {params['v_threshold']}")
            self.log_message(f"  Morphological Open Size: {params['morph_open_size']}")
            self.log_message(f"  Morphological Close Size: {params['morph_close_size']}")
            self.log_message(f"  Min Contour Area: {params['min_contour_area']}%")
            self.log_message(f"  Max Stray Pixel Area: {params['max_stray_area']}%")
            self.log_message(f"  Feather Type: {params['feather_type'] or 'None'}")
            self.log_message(f"  Feather Amount: {params['feather_amount']}")
            self.log_message(f"  Smoothing Type: {params['smoothing_type'] or 'None'}")
            self.log_message(f"  Smoothing Amount: {params['smoothing_amount']}")

            with tempfile.NamedTemporaryFile(suffix='.png', delete=False) as temp_file:
                temp_output = temp_file.name

            try:
                image = cv2.imread(self.sample_image_path)
                if image is None:
                    raise ValueError("Could not load image with OpenCV")

                # Resize image for faster preview
                max_preview_size = 800
                if max(image.shape[:2]) > max_preview_size:
                    scale = max_preview_size / max(image.shape[:2])
                    image = cv2.resize(image, None, fx=scale, fy=scale, interpolation=cv2.INTER_AREA)
                    with tempfile.NamedTemporaryFile(suffix='.png', delete=False) as temp_input:
                        cv2.imwrite(temp_input.name, image)
                        input_path = temp_input.name
                else:
                    input_path = self.sample_image_path

                bg_remover = BackgroundRemover({
                    'bg_remove_method': 3,
                    'params': params,
                    'source_path': input_path,
                    'output_path': temp_output
                })

                result = bg_remover.remove_background(input_path, temp_output)

                if result and os.path.exists(temp_output):
                    processed_pixmap = QPixmap(temp_output)
                    if not processed_pixmap.isNull():
                        self.preview_item.setPixmap(processed_pixmap)
                        self.graphics_scene.setSceneRect(self.preview_item.boundingRect())
                        self.log_message("Preview updated successfully")
                    else:
                        self.log_message("Error: Processed image could not be loaded into QPixmap", error=True)
                        self.preview_item.setPixmap(QPixmap(self.sample_image_path))
                else:
                    self.log_message("Error: Background removal failed", error=True)
                    self.preview_item.setPixmap(QPixmap(self.sample_image_path))

                # Clean up temporary files
                QTimer.singleShot(1000, lambda: os.unlink(temp_output))
                if input_path != self.sample_image_path:
                    QTimer.singleShot(1000, lambda: os.unlink(input_path))

            except Exception as e:
                self.log_message(f"Error during background removal: {str(e)}", error=True)
                self.log_message(traceback.format_exc(), error=True)
                self.preview_item.setPixmap(QPixmap(self.sample_image_path))

        except Exception as e:
            self.log_message(f"Error updating preview: {str(e)}", error=True)
            self.log_message(traceback.format_exc(), error=True)
            self.preview_item.setPixmap(QPixmap(self.sample_image_path))
            
    def create_parameter_widgets(self):
        """Create parameter widgets for Method 3"""
        while self.param_layout.count():
            item = self.param_layout.takeAt(0)
            if item.widget():
                item.widget().deleteLater()
            elif item.layout():
                while item.layout().count():
                    sub_item = item.layout().takeAt(0)
                    if sub_item.widget():
                        sub_item.widget().deleteLater()
        
        self.param_widgets = {}
        form_layout = QFormLayout()
        form_layout.setFormAlignment(Qt.AlignLeft | Qt.AlignTop)
        form_layout.setLabelAlignment(Qt.AlignLeft)
        
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
        
        # Morphological open size
        open_spin = QSpinBox()
        open_spin.setRange(1, 15)
        open_spin.setValue(3)
        form_layout.addRow("Morph Open Size:", open_spin)
        self.param_widgets['morph_open_size'] = open_spin
        
        # Morphological close size
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
        
        # Maximum stray pixel area
        stray_spin = QDoubleSpinBox()
        stray_spin.setRange(0.001, 0.5)
        stray_spin.setSingleStep(0.005)
        stray_spin.setValue(0.01)
        form_layout.addRow("Max Stray Pixel Area (%):", stray_spin)
        self.param_widgets['max_stray_area'] = stray_spin
        
        # Feathering controls
        feather_combo = QComboBox()
        feather_combo.addItems(["None", "Gaussian", "Bilateral", "Morphological"])
        feather_combo.setCurrentIndex(1)
        form_layout.addRow("Feathering Type:", feather_combo)
        self.param_widgets['feather_type'] = feather_combo
        
        feather_slider = QSlider(Qt.Horizontal)
        feather_slider.setRange(0, 20)
        feather_slider.setValue(5)
        feather_slider.setTickPosition(QSlider.TicksBelow)
        feather_slider.setTickInterval(1)
        form_layout.addRow("Feathering Amount:", feather_slider)
        self.param_widgets['feather_amount'] = feather_slider
        
        feather_value = QLabel("5")
        feather_value.setAlignment(Qt.AlignCenter)
        form_layout.addRow("", feather_value)
        feather_slider.valueChanged.connect(lambda v: feather_value.setText(str(v)))
        
        # Smoothing controls
        smoothing_combo = QComboBox()
        smoothing_combo.addItems(["None", "Median", "Gaussian"])
        smoothing_combo.setCurrentIndex(0)
        form_layout.addRow("Smoothing Type:", smoothing_combo)
        self.param_widgets['smoothing_type'] = smoothing_combo
        
        smoothing_slider = QSlider(Qt.Horizontal)
        smoothing_slider.setRange(0, 15)
        smoothing_slider.setValue(0)
        smoothing_slider.setTickPosition(QSlider.TicksBelow)
        smoothing_slider.setTickInterval(1)
        form_layout.addRow("Smoothing Amount:", smoothing_slider)
        self.param_widgets['smoothing_amount'] = smoothing_slider
        
        smoothing_value = QLabel("0")
        smoothing_value.setAlignment(Qt.AlignCenter)
        form_layout.addRow("", smoothing_value)
        smoothing_slider.valueChanged.connect(lambda v: smoothing_value.setText(str(v)))
        
        self.param_layout.addLayout(form_layout)
        
        # Connect signals for immediate preview updates
        for key, widget in self.param_widgets.items():
            if isinstance(widget, (QSpinBox, QDoubleSpinBox)):
                widget.valueChanged.connect(self.queue_preview_update)
            elif isinstance(widget, QComboBox):
                widget.currentIndexChanged.connect(self.queue_preview_update)
            elif isinstance(widget, QSlider):
                widget.valueChanged.connect(self.queue_preview_update)
 
    def process_backgrounds(self):
        """Start background removal processing"""
        self.clear_log()
        self.log_message("Starting background removal...")
        
        if not BACKGROUND_REMOVAL_AVAILABLE:
            QMessageBox.critical(self, "Missing Module",
                                "The BackgroundRemover module is not available in your installation.")
            self.log_message("Error: The BackgroundRemover module is not available", error=True)
            return
        
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
        
        self.save_bg_settings()
        self.log_message("Background removal settings saved")
        
        params = {
            's_threshold': self.param_widgets.get('s_threshold', QSpinBox()).value(),
            'v_threshold': self.param_widgets.get('v_threshold', QSpinBox()).value(),
            'morph_open_size': self.param_widgets.get('morph_open_size', QSpinBox()).value(),
            'morph_close_size': self.param_widgets.get('morph_close_size', QSpinBox()).value(),
            'min_contour_area': self.param_widgets.get('min_contour_area', QDoubleSpinBox()).value(),
            'max_stray_area': self.param_widgets.get('max_stray_area', QDoubleSpinBox()).value(),
            'feather_type': {0: None, 1: 'gaussian', 2: 'bilateral', 3: 'morph'}.get(
                self.param_widgets.get('feather_type', QComboBox()).currentIndex(), None
            ),
            'feather_amount': self.param_widgets.get('feather_amount', QSlider()).value(),
            'smoothing_type': {0: None, 1: 'median', 2: 'gaussian'}.get(
                self.param_widgets.get('smoothing_type', QComboBox()).currentIndex(), None
            ),
            'smoothing_amount': self.param_widgets.get('smoothing_amount', QSlider()).value()
        }
        
        bg_settings = {
            'source_path': source_path,
            'output_path': output_path,
            'bg_remove_method': 3,
            'params': params
        }
        
        self.log_message("HSV Adaptive Contour Extraction")
        self.log_message("Parameters:")
        self.log_message(f"  Saturation Threshold: {params['s_threshold']}")
        self.log_message(f"  Value Threshold: {params['v_threshold']}")
        self.log_message(f"  Morphological Open Size: {params['morph_open_size']}")
        self.log_message(f"  Morphological Close Size: {params['morph_close_size']}")
        self.log_message(f"  Min Contour Area: {params['min_contour_area']}%")
        self.log_message(f"  Max Stray Pixel Area: {params['max_stray_area']}%")
        self.log_message(f"  Feather Type: {params['feather_type'] or 'None'}")
        self.log_message(f"  Feather Amount: {params['feather_amount']}")
        self.log_message(f"  Smoothing Type: {params['smoothing_type'] or 'None'}")
        self.log_message(f"  Smoothing Amount: {params['smoothing_amount']}")
        
        self.progress_bar.setVisible(True)
        self.progress_bar.setValue(0)
        
        self.setEnabled(False)
        
        try:
            from processing.background_remover import BackgroundRemover
            self.bg_remover = BackgroundRemover(bg_settings)
            self.bg_remover.progress_update.connect(self.update_progress)
            self.bg_remover.processing_complete.connect(self.bg_processing_finished)
            self.bg_remover.error_occurred.connect(self.handle_error)
            self.bg_remover.log_message.connect(self.log_message)
            self.bg_remover.start()
        except Exception as e:
            QMessageBox.critical(self, "Error", 
                                f"Failed to initialize BackgroundRemover: {str(e)}")
            self.log_message(f"Error: Failed to initialize BackgroundRemover: {str(e)}", error=True)
            self.setEnabled(True)
            self.progress_bar.setVisible(False)
        
    def load_bg_settings(self):
        """Load saved settings for Background Removal tab and clean up old settings"""
        self.bg_source_input.setText(self.settings.value("bg_source_path", ""))
        self.bg_output_input.setText(self.settings.value("bg_output_path", ""))
        self.settings.remove("bg_remove_method")

    def save_bg_settings(self):
        """Save Background Removal settings"""
        self.settings.setValue("bg_source_path", self.bg_source_input.text())
        self.settings.setValue("bg_output_path", self.bg_output_input.text())

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

    def bg_processing_finished(self, success):
        """Handle completion of background removal processing"""
        self.setEnabled(True)
        self.progress_bar.setVisible(False)
        
        if success:
            QMessageBox.information(self, "Complete", "Background removal completed successfully!")
        else:
            QMessageBox.warning(self, "Warning", "Background removal completed with warnings or errors. Check the log for details.")
    
    