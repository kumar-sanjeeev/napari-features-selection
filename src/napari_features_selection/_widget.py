
"""
This module is an example of a barebones QWidget plugin for napari
It implements the Widget specification.
see: https://napari.org/stable/plugins/guides.html?#widgets
Replace code below according to your needs.
"""
from random import choices
import warnings
import numpy as np
import pandas as pd
# from magicgui.widgets import create_widget
from typing import TYPE_CHECKING
from magicgui import magic_factory
from qtpy.QtWidgets import QHBoxLayout, QPushButton, QWidget, QLabel, QVBoxLayout, QFileDialog

import os
import pandas as pd
#imports for file selection
# import tkinter as tk
# from tkinter import filedialog

from napari import Viewer
from pathlib import Path

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
        btn1 = QPushButton("Select")
        btn1.clicked.connect(self.read_content)
        
        # btn2 = QPushButton("Print Path")
        # btn2.clicked.connect(self.read_content)


        path_selection_container = QWidget()
        path_selection_container.setLayout(QHBoxLayout())

        path_selection_container.layout().addWidget(QLabel("File Path"))
        path_selection_container.layout().addWidget(btn1)
        # path_selection_container.layout().addWidget(btn2)
        self.layout().addWidget(title_container)
        self.layout().addWidget(path_selection_container)
    
    def path_selection(self):
        # root = tk.Tk()
        # root.withdraw()
        # allowed_file_types = ('*.csv')
        # path = filedialog(title='Choose file', filetypes = allowed_file_types)
        # print("Selected file path is: ", path)
        # print("Function call successful")
        file_filter = '*.csv'
        response = QFileDialog.getOpenFileName(parent= self, caption = 'Select a file', directory= os.getcwd(), filter=file_filter)
        print("Printing from path_selection function: ",response[0])
        return response
    
    def read_content(self, path):
        path,_ = self.path_selection()
        print("Printing from read_content function: ", path)
        df = pd.read_csv(path)
        print(df)

#need to find out its utility
@magic_factory
def example_magic_widget(img_layer: "napari.layers.Image"):
    pass

"""
IMPROVED GUI VERSION
"""
def _init_classifier(widget):
    """
    Classifier Initialization Widget initialization
    Parameters
    ----------
    widget: napari widget
    """
    print("This will run whenever I will select the widget from plugin window")

    def get_feature_choices(*args):
        """
        Function loading the column names of the widget dataframe
        """
        try:
            dataframe = pd.read_csv(widget.feature_path.value)
            return list(dataframe.columns)
        except IOError:
            return [""]
        
    widget.feature_selection._default_choices = get_feature_choices

    @widget.feature_path.changed.connect
    def update_df_columns():
        """
        Handles updating of dropdown options and setting defaults
        """
        # ...reset_choices() calls the "get_feature_choices" function above
        # to keep them updated with the current dataframe
        widget.feature_selection.reset_choices()


@magic_factory(
    feature_path={"label": "File Path:", "filter": "*.csv"},feature_selection={"choices": [""],"allow_multiple": True,"label": "Input Features:",},widget_init=_init_classifier,call_button="Run Feature Selection")
def initialize_classifier(viewer: Viewer,feature_path: Path,feature_selection=[""],):
    pass
