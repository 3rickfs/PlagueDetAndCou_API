"""Module to provide a toolbox with AI algorithms in order to make predictions
about detecting and couting radon222 marks on images
"""
from config import *
import sys
import os
from abc import ABC, abstractmethod
import cv2
class DetCouPredOperations():
    """Call this class to build an interface to prediction operation classes
    """

    @abstractmethod
    def operation(self,**kwargs):
        pass

class get_radon_image(DetCouPredOperations):
    """Call this class and return the radon image"""

    def operation(self,**kwargs):
        pred_dict = kwargs
        pred_dict["radon_image_name"] = "testimage1.jpg"
        return pred_dict

    def __str__():
        return "radon_img"

class make_detection_prediction(DetCouPredOperations):
    model = modellib.MaskRCNN(mode="inference", config=inference_config,  model_dir='logs')
    def __init__(self):
        model_path = os.path.join('logs', model_filename)
        self.model.load_weights(model_filename, by_name=True)

    def operation(self,**kwargs):
        pred_dict = kwargs
        result=self.model.detect([cv2.imread(pred_dict['input_img_path'])],verbose=1)[0]
        
        return {
            "tot_pa":len(result['class_name']),
            "tot_pb":len(result['class_name'])
        }
    def __str__():
        return "dete_pred"

class make_couting_operation(DetCouPredOperations):
    """Call this class to make a couting of radon marks
    and return a dictionary getting the img, radon222 marks coordinates and
    the number thereof as well as an error and a message
    """

    def operation(self,**kwargs):
        pred_dict = kwargs
        pred_dict["coun_pred"] = 20
        #pred_dict["error"] = 3
        #pred_dict["message"] = "Error en make_couting_operation"

        return pred_dict

    def __str__():
        return "coun_pred"

class DetCouPredProcedure:
    """Call this class to perform all prediction operations and return a
    dictionary in which there are elements as image path, detection preds,
    number of radon marks, error and message if necessary 
    """

    @staticmethod
    def call_operations(self,**kwargs):
        for operation in DetCouPredOperations.__subclasses__():
            kwargs = operation.operation(**kwargs)
        return {
            "tot_pa":kwargs["tot_pa"],
            "tot_pb":kwargs["tot_pb"]
        }

class RadonDetCouModel():
    """This function should be called from view module to get the prediction
    dictionary
    """

    def __init__(self):
        self.predictions_dict = {}

    def GetRadonDetCou(self, url):
        """Call this functions to make predictions """
        predictions = DetCouPredProcedure.call_operations(radon_image_path=url)

        return predictions


