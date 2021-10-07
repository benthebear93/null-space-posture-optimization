import tensorflow as tf
from tensorflow.keras.datasets import mnist
from tensorflow.keras.models import load_model
import autokeras as ak
import numpy as np
import pandas as pd
import sys

class StiffEstimator():
    def __init__(self, filename, model_name, axis, arg):
        self.filename = filename
        self.model_name = model_name
        self.axis_name = axis
        self.predict_stiff = []
        self.column_len = 0
        self.load_model()
        self.prediction(arg)

    def load_model(self):
        self.df2=pd.read_excel(self.filename)
        self.column_len = len(self.df2.columns)
        print("column_len", self.column_len)
        self.joint_data = self.df2[['J1','J2','J3','J4','J5','J6']]
    
    def prediction(self, arg):
        loaded_model = load_model(self.model_name, custom_objects=ak.CUSTOM_OBJECTS)
        self.predict_stiff = loaded_model.predict(tf.expand_dims(self.joint_data, -1))
        self.df2.insert(self.column_len, self.axis_name + "_"+arg+"_stiffness", self.predict_stiff)
        self.df2.to_excel(self.filename, index=False, float_format="%.3f", header=True)

# for i in range(len(predicted_z)):
#     if i!=0 and i % 4 ==0:
#         print(" ")
#     print(predicted_z[i])

if __name__ == "__main__":
    arg = sys.argv
    if len(arg) <2:
        print("[Error] : No argument given, try 'non' or 'opt' as argument")
        sys.exit(1)
        
    model_name =["x_stiffness", "y_stiffness", "z_stiffness"]
    if arg[1] == 'opt':
        x_stiffness = StiffEstimator('optimized_result.xlsx', model_name[0], "x", arg[1])
        y_stiffness = StiffEstimator('optimized_result.xlsx', model_name[1], "y", arg[1])
        z_stiffness = StiffEstimator('optimized_result.xlsx', model_name[2], "z", arg[1])
    else:
        x_stiffness = StiffEstimator('non_optimized_result.xlsx', model_name[0], "x", arg[1])
        y_stiffness = StiffEstimator('non_optimized_result.xlsx', model_name[1], "y", arg[1])
        z_stiffness = StiffEstimator('non_optimized_result.xlsx', model_name[2], "z", arg[1])
    