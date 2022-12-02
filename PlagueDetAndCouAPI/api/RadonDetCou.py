"""Module to provide a toolbox with AI algorithms in order to make predictions
about detecting and couting radon222 marks on images
"""
import sys
import os
from abc import ABC, abstractmethod

import numpy as np
from config import *
from IA_ModelsToolBox import utils

class DetCouPredOperations():
    """Call this class to build an interface to prediction operation classes
    """

    @abstractmethod
    def operation(self,**kwargs):
        pass

class get_plague_image(DetCouPredOperations):
    """Call this class and return the plague image"""

    def operation(self,**kwargs):
        pred_dict = kwargs
        dim = (600,800) #MODIFICAR DE ACUERDO A LO QUE SE CONSIDERE NECESARIO
        img = utils.read_img(pred_dict["input_img_path"], dim)
        img = np.expand_dims(img, axis=0) #REVISAR SI ES NECESARIO AGREGAR UNA DIMENSION MAS ej: de 600x800x3 a 1x600x800x3
        pred_dict["plague_image"] = img
        
        return pred_dict

    def __str__():
        return "radon_img"
        
class load_model(DetCouPredOperations):
    """Load de model..."""
    
    def operation(**kwargs):
        pred_dict = kwargs
        pred_dict["model"] = modellib.MaskRCNN(mode="inference", 
                                               config=inference_config,
                                               model_dir='logs')
        
        pred_dict["model"].load_weights(pred_dict["model_path"], by_name=True)
        
        return pred_dict
    

class make_detection_prediction(DetCouPredOperations):
    """ Perform model predictions """
    
    def operation(**kwargs):
        pred_dict = kwargs
        result=pred_dict["model"].detect([pred_dict["plague_image"]], verbose=1)
        pred_dict["dete_pred"] = {"pred_1": [1,2,3,4],
                                  "pred_2": [4,3,2,1]
                                 }
        pred_dict["result"]=result[0]
        #pred_dict["error"] = 2
        #pred_dict["message"] = "Error en make_detection_prediction" 

        return pred_dict

    def __str__():
        return "dete_pred"

class make_couting_operation(DetCouPredOperations):
    """Perform plague couting operations"""

    def operation(self,**kwargs):
        #CONSIDERAR AQUI EL PROCESO DE CUENTAS TENIENDO EN CUENTA
        #LA VARIABLE pred_dict["result"] QUE ESTABLECISTE EN EL PASO ANTERIOR
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
    number of plagues, error and message if necessary 
    """

    @staticmethod
    def call_operations(self,**kwargs):
        predictions_dict = {}
        predictions_dict["radon_image_path"] = kwargs["radon_image_path"]
        predictions_dict["error"] = 0
        predictions_dict["message"] = "No hay error"

        for operation in DetCouPredOperations.__subclasses__():
            predictions_dict = operation.operation(**predictions_dict)
            if predictions_dict['error'] != 0:
                break

        return predictions_dict

