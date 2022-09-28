"""
This module is an example of a barebones QWidget plugin for napari
It implements the Widget specification.
see: https://napari.org/stable/plugins/guides.html?#widgets
Replace code below according to your needs.
"""

import warnings
import numpy as np
import pandas as pd
from typing import TYPE_CHECKING
from magicgui import magic_factory
from qtpy.QtWidgets import QHBoxLayout, QPushButton, QWidget, QLabel, QVBoxLayout, QFileDialog

from napari.utils.notifications import show_info

import os
from napari import Viewer
from pathlib import Path

from magicgui.widgets import Table,Select,PushButton

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




#### working on saving the output file options

class SaveFile(QWidget):
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
        title_container.layout().addWidget(QLabel("<b>Save Output CSV File</b>"))

        #for path selection widget and buttons
        btn1 = QPushButton("Select Directory")
        btn1.clicked.connect(self.path_selection)

        path_selection_container = QWidget()
        path_selection_container.setLayout(QHBoxLayout())

        path_selection_container.layout().addWidget(QLabel(" Output File Path"))
        path_selection_container.layout().addWidget(btn1)
        self.layout().addWidget(title_container)
        self.layout().addWidget(path_selection_container)
    
    def path_selection(self):
        file_filter = '*.txt'
        file_path = QFileDialog.getSaveFileName(parent=self, caption='Save a csv File', directory=os.getcwd(), filter=file_filter)
        print("Printing the selected output file path: ",file_path[0])

        with open(file_path[0], 'w') as f:
            content = "Successful implemented the output path selection feature"
            f.write(content)
    


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
    print("User Selected the GUI version 2.0")

    def get_feature_choices(*args):
        """
        Function loading the column names of the widget dataframe
        """
        try:
            dataframe = pd.read_csv(widget.file_path.value)
            return list(dataframe.columns)
        except IOError:
            return [""]

    widget.drop_features._default_choices = get_feature_choices   
    widget.target_variable._default_choices = get_feature_choices
    widget.table._default_choices = None

    """
    # Updating the file path when it changes in GUI
    # Updating the deafult value of feature_selection
    # Updating the deafult value of the target_variable
    """
    @widget.file_path.changed.connect
    def update_df_columns():
        # ...reset_choices() calls the "get_feature_choices" function above
        # to keep them updated with the current dataframe

        widget.drop_features.reset_choices()
        widget.target_variable.reset_choices()
        show_info(f"Selected Path: {widget.file_path.value}")

        #reading the contents
        df = pd.read_csv(widget.file_path.value)

        # Can we only show head of dataframe ?
        #df_head = df.head(10)

        df_to_list = df.to_dict(orient='list')
        widget.table.value = df_to_list

    """
    # Updating the target_varible value when it changes in GUI
    """
    @widget.target_variable.changed.connect
    def update_target_variable():
        target_variable = widget.target_variable.value

    """
    Getting the values of selected drop features
    """
    @widget.drop_features.changed.connect
    def get_selected_drop_features():
        drop_features = widget.drop_features.value
        return drop_features

    """
    Drop the selected features and update the dataframe table
    """
    @widget.drop.changed.connect
    def drop_features():
        features_to_drop = get_selected_drop_features()
        show_info(f"Droppping features{features_to_drop}")
        df = pd.read_csv(widget.file_path.value)
        widget.table.value = df.drop(features_to_drop,axis=1)


@magic_factory(
    file_path={"label": "File Path:", "filter": "*.csv"},
    table = {"widget_type": Table, "label": "Data frame", "value":None,"enabled":False},
    target_variable= {"choices":[""], "label": "Target Variable"},
    drop_features = {"widget_type":Select, "label":"Select Features to Drop", "choices":[""], "allow_multiple":True},
    drop ={"widget_type":PushButton,"text":" Drop Features", "value":False},
    widget_init=_init_classifier,call_button="Print GUI selected parameters (Temporary)",)
def initialize_classifier(viewer: Viewer,file_path = Path.home(),table = Table,target_variable = "",drop_features=[""],drop=PushButton(value=False),):
    print('----Current selected parameter varible----')
    print("File Path:" ,file_path)
    print("Target variable: ", target_variable)
    print("Features to Drop: ", drop_features)



# adding table view widget

# temp_data = pd.DataFrame({'a':["Sanjeev", "sanju"],
#                      'b':[100,200],
#                      'c':['a','b']})

# d = temp_data.to_dict(orient='list')


# dict_of_lists = {"col_1": [1, 4], "col_2": [2, 5], "col_3": [3, 6]}


# def _init_table_view(widget):
    # print("Select Table View widget")

    # widget.table.value = None
    # print(widget.table.row_headers)
    # print(widget.table.column_headers)
    # widget.table.column_headers = ("a","b","c")
    # print(widget.table.column_headers)

    #for drop_featurs
    # print("print selected option")
    # print(widget.my_test.value)

    # @widget.my_test.changed.connect
    # def update():
    #     print(widget.my_test.value)

    # @widget.push.changed.connect
    # def print_something():
    #     print("Pressing push button")


# @magic_factory(table = {"widget_type": Table, "value":None,"label":"Dataframe"}, widget_init=_init_table_view,)
# def table_view(table):
#     print(table)
#     pass

# options = ['f1','f2','f3']
# @magic_factory(my_test = {"widget_type":Select, "label":'Drop Features', "choices":[""],"allow_multiple":True}, widget_init=_init_table_view)
# def table_view(my_test):
#     pass

# @magic_factory(push = {"widget_type":PushButton,"text":"Please click","enabled":False}, call_button=False,widget_init=_init_table_view)
# def table_view(push):
#     pass

