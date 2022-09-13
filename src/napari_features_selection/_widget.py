"""
This module is an example of a barebones QWidget plugin for napari

It implements the Widget specification.
see: https://napari.org/stable/plugins/guides.html?#widgets

Replace code below according to your needs.
"""
import numpy as np
import pandas as pd
# from magicgui.widgets import create_widget
from typing import TYPE_CHECKING
from magicgui import magic_factory
from qtpy.QtWidgets import QHBoxLayout, QPushButton, QWidget, QLabel, QVBoxLayout

if TYPE_CHECKING:
    import napari

class FeaturesSelection(QWidget):
    # your QWidget.__init__ can optionally request the napari viewer instance
    # in one of two ways:
    # 1. use a parameter called `napari_viewer`, as done here
    # 2. use a type annotation of 'napari.viewer.Viewer' for any parameter
    def __init__(self, napari_viewer):
        super().__init__()

        self.viewer = napari_viewer
        self.create_layout()

    
    def create_layout(self):
        self.worker = None
        self.setLayout(QVBoxLayout())
        #for title 
        title_container = QWidget()
        title_container.setLayout(QVBoxLayout())
        title_container.layout().addWidget(QLabel("<b>Input Features CSV File</b>"))

        #for path selection widget and buttons
        path_selection_container = QWidget()
        path_selection_container.setLayout(QHBoxLayout())
        btn = QPushButton("Select")
        path_selection_container.layout().addWidget(QLabel("File Path"))
        path_selection_container.layout().addWidget(btn)

        self.layout().addWidget(title_container)
        self.layout().addWidget(path_selection_container)


#need to find out its utility
@magic_factory
def example_magic_widget(img_layer: "napari.layers.Image"):
    pass
