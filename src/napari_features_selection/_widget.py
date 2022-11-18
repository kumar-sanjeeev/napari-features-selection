"""
This module is written for creation of GUI for features selection plugin in napari
"""

import os
import warnings
import numpy as np
import pandas as pd
from typing import TYPE_CHECKING
from magicgui import magic_factory
from napari.utils.notifications import show_info
from napari import Viewer
from pathlib import Path
from magicgui.widgets import Table,Select,PushButton, Slider, Label,FileEdit,FloatSlider
from ._ga_feature_selection import FeatureSelectionGA

if TYPE_CHECKING:
    import napari


def _init_widget(widget):
    """Widget initialization function.
    
    Args:
        widget (napari widget) : instance of ``FunctionGui``.
    """
    show_info(" Feature Selection plugin initialized in Napari")

    def get_feature_choices(*args):
        """
        Loading the column names of the input dataframe.

        Args:
            *args: variable length argument list.
        """
        try:
            dataframe = pd.read_csv(widget.file_path.value)
            return list(dataframe.columns)
        except IOError:
            return [""]

    widget.drop_features._default_choices = get_feature_choices   
    widget.target_variable._default_choices = get_feature_choices
    widget.table.value = None    

    @widget.reset.changed.connect
    def reset_arguments():
        """Resetting all GUI parameters"""
        show_info("Resetting the arguments to default values")
        widget.crossover_prob.value = 0.1
        widget.population_size.value = 10
        widget.generations.value = 5
        widget.output_file.value = ""
        widget.file_path.value = ""
        widget.drop_features.choices = [""]
        widget.target_variable.choices = [""]

 
    @widget.file_path.changed.connect
    def update_df_columns():
        """ Updating target varible, drop features and table widget in the gui, when input file path changes.
        
        Raises:
            Assertion Error: Either input file is not selected or its extension is not ``.csv``.

        """
        widget.table.value = None
        # input file extension check
        file_name, file_extension = os.path.splitext(widget.file_path.value)
        assert file_extension==".csv", "Either no file is selected or selected file type is not csv"

        # ...reset_choices() calls the "get_feature_choices" function above
        # to keep them updated with the current dataframe
        show_info(f"Selected Path: {widget.file_path.value}")
        widget.drop_features.reset_choices()
        widget.target_variable.reset_choices()
        
        # reading and updating the dataframe widget in GUI
        df = pd.read_csv(widget.file_path.value)
        df_head = df.head(20)
        df_to_list = df_head.to_dict(orient='list')
        widget.table.value = df_to_list


    
    @widget.target_variable.changed.connect
    def update_target_variable():
        """Updating the target_varible value when changes in GUI."""
        target_variable = widget.target_variable.value

  
    @widget.drop_features.changed.connect
    def get_selected_drop_features():
        """Getting names of the features selected to drop from GUI."""
        drop_features = widget.drop_features.value
        return drop_features

   
    @widget.drop.changed.connect
    def drop_features():
        """Dropping selected features from GUI."""
        features_to_drop = get_selected_drop_features()
        show_info(f"Droppping features{features_to_drop}")
        df = pd.read_csv(widget.file_path.value)
        widget.table.value = df.drop(features_to_drop,axis=1)

        # updating target variable available choices
        widget.target_variable.choices = set(widget.drop_features.choices) - set(features_to_drop)

    
    @widget.output_file.changed.connect
    def update_output_filepath():
        """Updating the output file location."
        
        Raises:
            Assertion Error: If output file extension is not ``.csv``.

        """
        file_path = widget.output_file.value
        _, file_ext = os.path.splitext(file_path)
        assert file_ext==".csv", f"Wrong file extension, acceptable extension is .csv, but user provided: {file_ext}" 
        show_info(f"Selected File Saving location: {os.path.basename(file_path)}")    


    @widget.generations.changed.connect
    def update_generations():
        """Updating the generation value when changes in GUI."""
        generations = widget.generations.value

   
    @widget.population_size.changed.connect
    def update_population_size():
        """Updating the population size value when changes in GUI."""
        population_size = widget.population_size.value
      
    @widget.crossover_prob.changed.connect
    def update_crossover_prob():
        """Updating the cross over probability value when changes in GUI."""
        crossover_prob = widget.crossover_prob.value
    

    @widget.run_ga.changed.connect
    def run_ga():
        """Runs GA for feature selection.
        
        Raises:
            Assertion Error: If output file location is not selected.
            Assertion Error: If someone runs the GA without selecting input file
        """
        out_file_path = widget.output_file.value
        _, file_ext = os.path.splitext(out_file_path)
        assert file_ext==".csv", f"Output file location is: {file_ext}, please select the output file location" 
        in_file_path = widget.file_path.value

        _, file_ext = os.path.splitext(in_file_path)

        assert file_ext == ".csv", "Nothing to run, first select the input csv file"

        # Get input parameters required for GA class
        Dataset = pd.read_csv(in_file_path)
        Target = widget.target_variable.value
        Drop_features = widget.drop_features.value
        Crossover_prob = widget.crossover_prob.value
        Population_size = widget.population_size.value
        Generation = widget.generations.value
        Out_dir = widget.output_file.value
        Max_features = None

        obj = FeatureSelectionGA(file_path=in_file_path, target=Target, drop_features=Drop_features)
        clf, X_train, X_test, y_train, y_test, acc = obj.process_data()

        obj.run_GA(generations= Generation, population_size = Population_size, 
                   crossover_probability= Crossover_prob, max_features=None, outdir= Out_dir, 
                   classifier=clf,X_train_trans= X_train, X_test_trans= X_test, 
                   y_train = y_train, y_test= y_test)

        show_info("GA feature selection completed.")
        
"""magic factory is decorator, which, will autogenerate a graphical user interface(GUI)"""  
@magic_factory(
    reset = {"widget_type":PushButton, "text":" Reset Plugin Arguments", "value":False},
    file_path={"widget_type":FileEdit,"mode":'r',"label": "File Path:", "filter": "*.csv"},
    table = {"widget_type":Table, "label": "Data frame", "value":None,"enabled":True},
    drop_features = {"widget_type":Select, "label":"Select Features to Drop", "choices":[""], "allow_multiple":True},
    drop ={"widget_type":PushButton,"text":" Drop Features", "value":False},
    target_variable= {"label": "Target Variable", "choices":[""]},
    widget_init=_init_widget,
    output_file= {"widget_type":FileEdit, "mode":'w', "filter":"*.csv"},
    label= {"widget_type":Label,"label":" HyperParameters for Genetic Algorithm","value":""},
    generations ={"widget_type":Slider, "label":"Number of Generations","max":20,"value":5},
    population_size={"widget_type":Slider, "label":"Population Size","max":100,"value":10},
    crossover_prob ={"widget_type":FloatSlider, "label":"Crossover Probability","max":1,"value":0.1},
    run_ga = {"widget_type":PushButton, "text": "Run GA Feature Selection", "value":False},
    call_button="[For Info] Print Selected GUI parameters",
)
def initialize_widget(
    viewer: Viewer,
    reset = PushButton(value=False),
    file_path = Path.home(),
    table = Table,
    drop_features=[""],
    drop=PushButton(value=False),
    target_variable = "",
    output_file=Path.home(),
    label=Label,
    generations=Slider(value=5),
    population_size=Slider(value=10),
    crossover_prob=Slider(value=0.1),
    run_ga=PushButton(value=False)):
    """Initialization function used to initialize the widget"""
    print("Input File path: ", file_path)
    print("Target variable: ", target_variable)
    print("Drop features: ", drop_features)
    print("Generations: ", generations)
    print("Population Size: ", population_size)
    print("Crossover Probability: ", crossover_prob)
    print("Output file path: ", output_file)
