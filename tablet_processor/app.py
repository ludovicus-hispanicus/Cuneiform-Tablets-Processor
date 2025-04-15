#!/usr/bin/env python3
"""
Cuneiform Tablet Processor
--------------------------
A Python-based application for processing and compositing
cuneiform tablet photographs.

This application provides functionality similar to the original Photoshop script,
including image composition, automatic scaling, and metadata embedding.
"""

import sys
import os
from PyQt5.QtWidgets import QApplication
from PyQt5.QtCore import Qt
from gui.main_window import MainWindow

def main():
    """Main application entry point"""
    # Create the application
    app = QApplication(sys.argv)
    app.setApplicationName("Cuneiform Tablet Processor")
    app.setOrganizationName("TabletProcessor")
    
    # Set application style
    app.setStyle("Fusion")
    
    # Create and show the main window
    window = MainWindow()
    window.show()
    
    # Run the application
    sys.exit(app.exec_())


if __name__ == "__main__":
    main()