#!/usr/bin/env python3
"""
Settings module for the Cuneiform Tablet Processor.
Handles loading, saving, and validating configuration.
"""

import os
import json
from PyQt5.QtCore import QSettings

class ProcessorSettings:
    """Class to manage application settings"""
    
    def __init__(self):
        """Initialize settings with default values"""
        self.settings = QSettings("TabletProcessor", "CuneiformApp")
        self.load_from_qsettings()
    
    def load_from_qsettings(self):
        """Load settings from QSettings"""
        self.source_path = self.settings.value("source_path", "")
        self.output_path = self.settings.value("output_path", "")
        
        self.add_logo = self.settings.value("add_logo", False, type=bool)
        self.logo_path = self.settings.value("logo_path", "")
        
        self.dpi = self.settings.value("dpi", 600, type=int)
        self.compression = self.settings.value("compression", "none")
        
        self.save_jpeg = self.settings.value("save_jpeg", False, type=bool)
        self.jpeg_quality = self.settings.value("jpeg_quality", 8, type=int)
        
        self.photographer = self.settings.value("photographer", "")
        self.institution = self.settings.value("institution", "")
        self.copyright_notice = self.settings.value("copyright_notice", "")
        self.usage_terms = self.settings.value("usage_terms", "")
        
        default_credit = (
            "Funding for photography and post-processing provided by a Sofja Kovalevskaja Award "
            "(Alexander von Humboldt Foundation, German Federal Ministry for Education and Research) "
            "as part of the Electronic Babylonian Literature-Projekt of the Ludwig-Maximilians-Universität München"
        )
        self.credit_line = self.settings.value("credit_line", default_credit)
    
    def save_to_qsettings(self):
        """Save settings to QSettings"""
        self.settings.setValue("source_path", self.source_path)
        self.settings.setValue("output_path", self.output_path)
        
        self.settings.setValue("add_logo", self.add_logo)
        self.settings.setValue("logo_path", self.logo_path)
        
        self.settings.setValue("dpi", self.dpi)
        self.settings.setValue("compression", self.compression)
        
        self.settings.setValue("save_jpeg", self.save_jpeg)
        self.settings.setValue("jpeg_quality", self.jpeg_quality)
        
        self.settings.setValue("photographer", self.photographer)
        self.settings.setValue("institution", self.institution)
        self.settings.setValue("copyright_notice", self.copyright_notice)
        self.settings.setValue("usage_terms", self.usage_terms)
        self.settings.setValue("credit_line", self.credit_line)
    
    def to_dict(self):
        """Convert settings to dictionary"""
        return {
            'source_path': self.source_path,
            'output_path': self.output_path,
            'add_logo': self.add_logo,
            'logo_path': self.logo_path,
            'dpi': self.dpi,
            'compression': self.compression,
            'save_jpeg': self.save_jpeg,
            'jpeg_quality': self.jpeg_quality,
            'photographer': self.photographer,
            'institution': self.institution,
            'copyright_notice': self.copyright_notice,
            'usage_terms': self.usage_terms,
            'credit_line': self.credit_line
        }
    
    def from_dict(self, settings_dict):
        """Load settings from dictionary"""
        self.source_path = settings_dict.get('source_path', self.source_path)
        self.output_path = settings_dict.get('output_path', self.output_path)
        
        self.add_logo = settings_dict.get('add_logo', self.add_logo)
        self.logo_path = settings_dict.get('logo_path', self.logo_path)
        
        self.dpi = settings_dict.get('dpi', self.dpi)
        self.compression = settings_dict.get('compression', self.compression)
        
        self.save_jpeg = settings_dict.get('save_jpeg', self.save_jpeg)
        self.jpeg_quality = settings_dict.get('jpeg_quality', self.jpeg_quality)
        
        self.photographer = settings_dict.get('photographer', self.photographer)
        self.institution = settings_dict.get('institution', self.institution)
        self.copyright_notice = settings_dict.get('copyright_notice', self.copyright_notice)
        self.usage_terms = settings_dict.get('usage_terms', self.usage_terms)
        self.credit_line = settings_dict.get('credit_line', self.credit_line)
    
    def validate(self):
        """Validate settings and return list of errors if any"""
        errors = []
        
        if not self.source_path:
            errors.append("Source path is not specified")
        elif not os.path.exists(self.source_path):
            errors.append(f"Source path does not exist: {self.source_path}")
            
        if not self.output_path:
            errors.append("Output path is not specified")
            
        if self.add_logo and not self.logo_path:
            errors.append("Logo is enabled but no logo file is specified")
        elif self.add_logo and not os.path.exists(self.logo_path):
            errors.append(f"Logo file does not exist: {self.logo_path}")
            
        if self.dpi not in [300, 600]:
            errors.append(f"Invalid DPI value: {self.dpi}")
            
        if self.compression not in ["none", "lzw", "zip"]:
            errors.append(f"Invalid compression type: {self.compression}")
            
        if self.save_jpeg and (self.jpeg_quality < 1 or self.jpeg_quality > 12):
            errors.append(f"Invalid JPEG quality: {self.jpeg_quality}")
            
        return errors