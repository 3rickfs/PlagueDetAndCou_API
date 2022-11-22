"""Module to provide a toolbox with AI algorithms in order to make predictions
about detecting and couting radon222 marks on images
"""

import sys
import os
from abc import ABC, abstractmethod

class DetCouPredOperations():
    """Call this class to build an interface to prediction operation classes
    """

    @abstractmethod
    def operation(**kwargs):
        pass

class get_radon_image(DetCouPredOperations):
    """Call this class and return the radon image"""

    def operation(**kwargs):
        pred_dict = kwargs
        pred_dict["radon_image_name"] = "testimage1.jpg"
        #pred_dict["error"] = 1
        #pred_dict["message"] = "Error en get_radon_img"

        return pred_dict

    def __str__():
        return "radon_img"

class make_detection_prediction(DetCouPredOperations):
    """Call this class to make a detection prediction of radon marks
    and return a dictionary getting the img and radon222 marks coordinates
    as well as the error and a message
    """

    def operation(**kwargs):
        pred_dict = kwargs
        pred_dict["dete_pred"] = {"pred_1": [1,2,3,4],
                                  "pred_2": [4,3,2,1],
                                 }
        #pred_dict["error"] = 2
        #pred_dict["message"] = "Error en make_detection_prediction" 

        return pred_dict

    def __str__():
        return "dete_pred"

class make_couting_operation(DetCouPredOperations):
    """Call this class to make a couting of radon marks
    and return a dictionary getting the img, radon222 marks coordinates and
    the number thereof as well as an error and a message
    """

    def operation(**kwargs):
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
    def call_operations(**kwargs):
        predictions_dict = {}
        predictions_dict["radon_image_path"] = kwargs["radon_image_path"]
        predictions_dict["error"] = 0
        predictions_dict["message"] = "No hay error"

        for operation in DetCouPredOperations.__subclasses__():
            predictions_dict = operation.operation(**predictions_dict)
            if predictions_dict['error'] != 0:
                break

        return predictions_dict

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


