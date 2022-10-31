"""
This module is an example of a barebones QWidget plugin for napari
It implements the Widget specification.
see: https://napari.org/stable/plugins/guides.html?#widgets
Replace code below according to your needs.
"""

import os
from random import random
import warnings
from matplotlib.dates import datestr2num
import numpy as np
import pandas as pd
from typing import TYPE_CHECKING
from magicgui import magic_factory, widgets
from qtpy.QtWidgets import QHBoxLayout, QPushButton, QWidget, QLabel, QVBoxLayout, QFileDialog

from napari.utils.notifications import show_info, show_error, show_console_notification

import os
from napari import Viewer
from pathlib import Path

from magicgui.widgets import Table,Select,PushButton, Slider, Label,FileEdit,FloatSlider, RangeEdit
import csv

from sklearn import preprocessing
from sklearn.ensemble import GradientBoostingClassifier
from sklearn.metrics import accuracy_score
from sklearn.model_selection import train_test_split
from sklearn_genetic import GAFeatureSelectionCV
from sklearn_genetic.callbacks import ProgressBar
import matplotlib.pyplot as plt


# import GA file
# from ._reader import napari_get_reader
from ._ga_feature_selection import FeatureSelectionGA

if TYPE_CHECKING:
    import napari

# class FeaturesSelection(QWidget):
#     # your QWidget.__init__ can optionally request the napari viewer instance
#     # in one of two ways:
#     # 1. use a parameter called `napari_viewer`, as done here
#     # 2. use a type annotation of 'napari.viewer.Viewer' for any parameter
#     def __init__(self, napari_viewer):
#         super().__init__()

#         self.viewer = napari_viewer
#         self.create_layout()

    
#     def create_layout(self):
#         self.worker = None
#         self.setLayout(QVBoxLayout())
#         #for title 
#         title_container = QWidget()
#         title_container.setLayout(QVBoxLayout())
#         title_container.layout().addWidget(QLabel("<b>Input Features CSV File</b>"))

#         #for path selection widget and buttons
#         btn1 = QPushButton("Select")
#         btn1.clicked.connect(self.read_content)
        
#         # btn2 = QPushButton("Print Path")
#         # btn2.clicked.connect(self.read_content)


#         path_selection_container = QWidget()
#         path_selection_container.setLayout(QHBoxLayout())

#         path_selection_container.layout().addWidget(QLabel("File Path"))
#         path_selection_container.layout().addWidget(btn1)
#         # path_selection_container.layout().addWidget(btn2)
#         self.layout().addWidget(title_container)
#         self.layout().addWidget(path_selection_container)
    
#     def path_selection(self):
#         # root = tk.Tk()
#         # root.withdraw()
#         # allowed_file_types = ('*.csv')
#         # path = filedialog(title='Choose file', filetypes = allowed_file_types)
#         # print("Selected file path is: ", path)
#         # print("Function call successful")
#         file_filter = '*.csv'
#         response = QFileDialog.getOpenFileName(parent= self, caption = 'Select a file', directory= os.getcwd(), filter=file_filter)
#         print("Printing from path_selection function: ",response[0])
#         return response
#     from distutils.errors import LibError

#     def read_content(self, path):
#         path,_ = self.path_selection()
#         print("Printing from read_content function: ", path)
#         df = pd.read_csv(path)
#         print(df)




#### working on saving the output file options

# class SaveFile(QWidget):
#     # your QWidget.__init__ can optionally request the napari viewer instance
#     # in one of two ways:
#     # 1. use a parameter called `napari_viewer`, as done here
#     # 2. use a type annotation of 'napari.viewer.Viewer' for any parameter
#     def __init__(self, napari_viewer):
#         super().__init__()

#         self.viewer = napari_viewer
#         self.create_layout()

    
#     def create_layout(self):
        
#         self.worker = None
#         self.setLayout(QVBoxLayout())
#         #for title 
#         title_container = QWidget()
#         title_container.setLayout(QVBoxLayout())
#         title_container.layout().addWidget(QLabel("<b>Save Output CSV File</b>"))

#         #for path selection widget and buttons
#         btn1 = QPushButton("Select Directory")
#         btn1.clicked.connect(self.path_selection)

#         path_selection_container = QWidget()
#         path_selection_container.setLayout(QHBoxLayout())

#         path_selection_container.layout().addWidget(QLabel(" Output File Path"))
#         path_selection_container.layout().addWidget(btn1)
#         self.layout().addWidget(title_container)
#         self.layout().addWidget(path_selection_container)
    
#     def path_selection(self):
#         file_filter = '*.txt'
#         file_path = QFileDialog.getSaveFileName(parent=self, caption='Save a csv File', directory=os.getcwd(), filter=file_filter)
#         print("Printing the selected output file path: ",file_path[0])

#         with open(file_path[0], 'w') as f:
#             content = "Successful implemented the output path selection feature"
#             f.write(content)
    


#need to find out its utility
# @magic_factory
# def example_magic_widget(img_layer: "napari.layers.Image"):
#     pass


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
    
    # """
    # Handling when save button is pressed
    # """

    # @widget.save.changed.connect
    # def update_save():
    #     file_path = update_output_filepath()
    #     show_info(f"Saving File {os.path.basename(file_path)}")

    #     # check the give file name extension is .csv, if yes, then only save
    #     file_name, file_ext = os.path.splitext(file_path)
    #     print(file_name)
    #     print(file_ext)

    #     assert file_ext==".csv", f"Wrong file extension, acceptable extension is .csv, but user provided: {file_ext}"

    #     dummy_df = pd.DataFrame({'a':["Sanjeev", "sanju"],
    #                  'b':[100,200],
    #                  'c':['a','b']})
    #     dummy_df.to_csv(file_path, index=False)

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
        
        # # process data
        # Label = Dataset[Target]
        # print(Label)
        # le = preprocessing.LabelEncoder()
        # y  = le.fit_transform(Label)

        # # Drop features
        # if Drop_features is not None:
        #     X = Dataset.drop(Drop_features, axis=1)
        #     X = X.drop(columns=[Target])
        # else:
        #     X = Dataset.drop(columns=[Target])


        # ## Transform Data
        # quantile_transformer = preprocessing.QuantileTransformer(random_state=0)
        # X_train, X_test, y_train, y_test = train_test_split(X,y, test_size=0.33, random_state=0)

        # X_train_trans = quantile_transformer.fit_transform(X_train)
        # X_test_trans = quantile_transformer.transform(X_test)

        # """Convert back to dataframes"""
        # X_train_trans = pd.DataFrame(X_train_trans, columns=X.columns)
        # X_test_trans = pd.DataFrame(X_test_trans, columns=X.columns)

        # """without GA feature selection"""
        # clf = GradientBoostingClassifier(n_estimators=10)
        # clf.fit(X_train_trans, y_train)
        # y_predict = clf.predict(X_test_trans)
        # accuracy_no_gs = accuracy_score(y_test, y_predict)

        # # print("Accuracy is," "{:.2f}".format(accuracy_no_gs))

        # """run GA algorithm"""
        # ga_estimator = GAFeatureSelectionCV(estimator=clf, 
        #                                     cv=5,
        #                                     scoring='accuracy',
        #                                     population_size=Population_size,
        #                                     generations=Generation,
        #                                     n_jobs=-1,
        #                                     crossover_probability=Crossover_prob,
        #                                     mutation_probability=0.05,
        #                                     verbose=True,
        #                                     max_features=Max_features,
        #                                     keep_top_k=3,
        #                                     elitism=True)

        # callback = ProgressBar()
        # ga_estimator.fit(X_train_trans, y_train, callbacks=callback)
        # features = ga_estimator.best_features_
        # y_predict_ga = ga_estimator.predict(X_test_trans.iloc[:,features])
        # accuracy = accuracy_score(y_test, y_predict_ga)
        # print("Accuracy with GA: ", accuracy)
        # print(ga_estimator.best_features_)

        # # plt.figure()
        # # plot = plot

        # selected_features = list(X_test_trans.iloc[:, features].columns)
        # print(selected_features)
        # # cv_results = ga_estimator.cv_results_

        # result_df = Dataset.loc[:,selected_features]
        # result_df.to_csv(Out_dir)
        # print(result_df)


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
    # output_file= {"widget_type":FileEdit, "mode":'w', "filter":"*.csv"},
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


# @magic_factory(filename={"widget_type":FileEdit,"mode":'w', "filter":"*.csv"},
#                save = {"widget_type":PushButton,"text":"Save File","value":False},
#                 widget_init=_init_table_view)
# def table_view(filename=Path.home(), save=PushButton(value=False)):
#     file_path = filename
#     print(pd.read_csv(file_path))
#     # with open(filename) as f:
#     #     lines = f.readlines()
#     #     print(lines)







# adding table view widget

# temp_data = pd.DataFrame({'a':["Sanjeev", "sanju"],
#                      'b':[100,200],
#                      'c':['a','b']})

# # d = temp_data.to_dict(orient='list')


# # dict_of_lists = {"col_1": [1, 4], "col_2": [2, 5], "col_3": [3, 6]}


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

    # print("Store the content of the files")
    
    # @widget.filename.changed.connect
    # def fchange():
    #     """
    #     will give me the path of the file that user has created when windows pop up
    #     """

    #     show_info(f"Selected file path: {widget.filename.value}")
        # basename = os.path.basename(widget.filename.value)
        # print(basename)
        # print(type(basename))
        # try:
        #     if (str(widget.filename.value).endswith(".csv")):
        #         return widget.filename.value
        #     else:
        # except IOError:
        #     print("something wrong")    

        # if not str(widget.filename.value).endswith(".csv"):

        #     show_error("Selected file name does not end with .csv")
        # return widget.filename.value


    
    # @widget.save.changed.connect
    # def save_file():
    #     print("Saving File")
    #     file_path = fchange()
    #     df = pd.DataFrame({'a':["Sanjeev", "sanju"],
    #                  'b':[100,200],
    #                  'c':['a','b']})
        
    #     df.to_csv(file_path, index=False)
        # csv_content = pd.DataFrame.to_csv(df, index=False)

        # with open(file_path, 'w') as f:
        #     writer = csv.writer(f)
        #     writer.writerow(csv_content)


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

# @magic_factory(filename={"widget_type":FileEdit,"mode":'w', "filter":"*.csv"},
#                save = {"widget_type":PushButton,"text":"Save File","value":False},
#                 widget_init=_init_table_view)
# def table_view(filename=Path.home(), save=PushButton(value=False)):
#     file_path = filename
#     print(pd.read_csv(file_path))
#     # with open(filename) as f:
#     #     lines = f.readlines()
#     #     print(lines)

# line_edit = LineEdit(value='hello')
# spin_box = SpinBox(value=400)
# @magic_factory(line_edit={"widget_type":LineEdit, "value":"Hello"}, button={"widget_type":PushButton, "value":False},layout='horizontal')
# def table_view(line_edit, button):
#     pass

# line_edit = LineEdit(value="Hello")
# button = PushButton(value=False)

# container = widgets.Container(widgets=[line_edit,button], layout="vertical", labels=False)

# @magic_factory(container={"widget_type":Container, "layout":'horizontal', "widgets":[]})
# def table_view(container=Container(widgets=[LineEdit(value="Hello"), PushButton(value=False)])):
#     pass





# class hyperparameters_Widgets:

#     def __init__(self, viewer) -> None:
#         pass
#         # self.gen = gen
#         # self.prob= prob
#         # self.size= size
#         self.viewer = viewer

#         widget = self.create_widget()

#         viewer.window.add_dock_widget(widget, area='right', name="hello")

#     def create_widget(self):
#         button_1 = widgets.PushButton(value= True, text='Save something')
#         button_2 = widgets.PushButton(value= False, text="ello")

#         container = widgets.Container(widgets=[button_1, button_2], layout='horizontal')


# @magic_factory(call_button="sss")
# def table_view(viewer: Viewer):
#     hyperparameters_Widgets(viewer)




# from magicgui import event_loop, magicgui
# from magicgui.widgets import Container


# class MyObject:
#     def __init__(self, name):
#         self.name = name
#         self.counter = 0.0

#     @magicgui(auto_call=True)
#     def method(self, sigma: float = 0):
#         print(f"instance: {self.name}, counter: {self.counter}, sigma: {sigma}")
#         self.counter = self.counter + sigma
#         return self.name


# @magic_factory()
