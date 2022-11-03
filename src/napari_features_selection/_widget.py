"""
This module is an example of a barebones QWidget plugin for napari
It implements the Widget specification.
see: https://napari.org/stable/plugins/guides.html?#widgets
Replace code below according to your needs.
"""

import os
import warnings
from matplotlib.dates import datestr2num
import numpy as np
import pandas as pd
from typing import TYPE_CHECKING
from magicgui import magic_factory

from napari.utils.notifications import show_info

import os
from napari import Viewer
from pathlib import Path

from magicgui.widgets import Table,Select,PushButton, Slider, Label,FileEdit,FloatSlider


from ._ga_feature_selection import FeatureSelectionGA

if TYPE_CHECKING:
    import napari

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

        # check whether selected file is csv, if not raise error
        file_name, file_extension = os.path.splitext(widget.file_path.value)
        assert file_extension==".csv", "Selected file type is not csv"

        #reading the contents
        df = pd.read_csv(widget.file_path.value)

        df_head = df.head(20)

        df_to_list = df_head.to_dict(orient='list')
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

    """
    Handling when output file path is changing
    """
    @widget.output_file.changed.connect
    def update_output_filepath():
         # check the give file name extension is .csv, if yes, then only save
        file_path = widget.output_file.value
        file_name, file_ext = os.path.splitext(file_path)
        assert file_ext==".csv", f"Wrong file extension, acceptable extension is .csv, but user provided: {file_ext}" 
        # show_info(f"Selected output file path is: {widget.output_file.value}")
        show_info(f"Saving File {os.path.basename(file_path)}")    

        dummy_df = pd.DataFrame({'a':["Sanjeev", "sanju"],
                     'b':[100,200],
                     'c':['a','b']})
        dummy_df.to_csv(file_path, index=False)  
        show_info(f"Saving File {os.path.basename(file_path)}")
        # return widget.output_file.value
    

    @widget.run_ga.changed.connect
    def run_ga():
        
        # final_data = pd.read_csv(widget.file_path.value)
        # read the input file and selected parameters to run the GA on features
        out_file_path = widget.output_file.value
        file_name, file_ext = os.path.splitext(out_file_path)
        assert file_ext==".csv", f"Please first select the output file location: {file_ext}" 
        # show_info(f"Selected output file path is: {widget.output_file.value}")
        # show_info(f"Saving File {os.path.basename(file_path)}") 

        in_file_path = widget.file_path.value

        _, file_ext = os.path.splitext(in_file_path)

        assert file_ext == ".csv", "Nothing to run, first select the input csv file"

        print("Selected csv file path: ", in_file_path)

        # Get input parameters required for GA class
        Dataset = pd.read_csv(in_file_path)
        Target = widget.target_variable.value
        Drop_features = widget.drop_features.value
        Crossover_prob = widget.crossover_prob.value
        Population_size = widget.population_size.value
        Generation = widget.generations.value
        Out_dir = widget.output_file.value
        Max_features = None
        
        print("Before Dropping Size is : ", Dataset.shape)

        obj = FeatureSelectionGA(file_path=in_file_path, target=Target, drop_features=Drop_features)
        clf, X_train, X_test, y_train, y_test, acc = obj.process_data()

        obj.run_GA(generations= Generation, population_size = Population_size, crossover_probability= Crossover_prob, max_features=None, outdir= Out_dir, classifier=clf,
                  X_train_trans= X_train, X_test_trans= X_test, y_train = y_train, y_test= y_test)
        show_info("GA feature selection completed")
        

    def update_generations():
        generations = widget.generations.value

    """
    Updating the population size when changed by slider
    """
    @widget.generations.changed.connect
    def update_population_size():
        population_size = widget.population_size.value
      
    """
    Updating the cross over prob value when changed by slider
    """
    @widget.crossover_prob.changed.connect
    def update_crossover_prob():
        crossover_prob = widget.crossover_prob.value
    
# save= {"widget_type":PushButton, "text": "Click to Save File", "value":False}
# , save=PushButton(value=False)
@magic_factory(
    file_path={"label": "File Path:", "filter": "*.csv"},
    table = {"widget_type": Table, "label": "Data frame", "value":None,"enabled":True},
    target_variable= {"choices":[""], "label": "Target Variable"},
    drop_features = {"widget_type":Select, "label":"Select Features to Drop", "choices":[""], "allow_multiple":True},
    drop ={"widget_type":PushButton,"text":" Drop Features", "value":False},
    widget_init=_init_classifier,
    output_file= {"widget_type":FileEdit, "mode":'w', "filter":"*.csv"},
    label= {"widget_type":Label,"label":" HyperParameters for Genertic Algorithm","value":""},
    generations ={"widget_type":Slider, "label":"Number of Generations","max":20,"value":5},
    population_size={"widget_type":Slider, "label":"Population Size","max":100,"value":10},
    crossover_prob ={"widget_type":FloatSlider, "label":"Crossover Probability","max":1,"value":0.1},
    run_ga = {"widget_type":PushButton, "text": "Run GA Feature Selection", "value":False},
    call_button=False,
)
def initialize_classifier(viewer: Viewer,file_path = Path.home(),table = Table,target_variable = "",drop_features=[""],drop=PushButton(value=False),output_file=Path.home(),label=Label,generations=Slider(value=5),population_size=Slider(value=10),crossover_prob=Slider(value=0.1),run_ga=PushButton(value=False)):
    print('----Current selected parameter varible----')
    print("File Path:" ,file_path)
    print("Target variable: ", target_variable)
    print("Features to Drop: ", drop_features)
    print("Generations",generations)
    print("Population Size", population_size)
    print("cross_over Prob", crossover_prob)

    print("reading data from output csv file: ", print(pd.read_csv(output_file)))


