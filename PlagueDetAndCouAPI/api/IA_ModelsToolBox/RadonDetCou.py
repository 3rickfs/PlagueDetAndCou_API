"""Module to provide a toolbox with AI algorithms in order to make predictions
about detecting and couting radon222 marks on images
"""
import sys
import os
from abc import ABC, abstractmethod
import cv2
import numpy as np
from IA_ModelsToolBox.config import *
from IA_ModelsToolBox import utils

class DetCouPredOperations():
    """Call this class to build an interface to prediction operation classes
    """

    @abstractmethod
    def operation(self,**kwargs):
        pass

class get_plague_image(DetCouPredOperations):
    """Call this class and return the plague image"""

    def operation(**kwargs):
        pred_dict = kwargs
        dim = (600,800) #MODIFICAR DE ACUERDO A LO QUE SE CONSIDERE NECESARIO
        img = utils.read_img(pred_dict['radon_image_path'], dim)
        #REVISAR SI ES NECESARIO AGREGAR UNA DIMENSION MAS ej: de 600x800x3 a 1x600x800x3
        pred_dict["plague_image"] = img
        return pred_dict

    def __str__():
        return "radon_img"
        
class load_model(DetCouPredOperations):
    """Load de model..."""
    
    def operation(**kwargs):
        pred_dict = kwargs
        pred_dict["model"] = modellib.MaskRCNN(mode="inference",config=inference_config,model_dir='logs')
        print(os.listdir('./'))
        model_path = os.path.join('logs',kwargs['aimodel_weights_path'])
        pred_dict["model"].load_weights(model_path, by_name=True)
        return pred_dict
    

class make_detection_prediction(DetCouPredOperations):
    """ Perform model predictions """
    
    def operation(**kwargs):
        pred_dict = kwargs
        result=pred_dict["model"].detect([pred_dict["plague_image"]], verbose=1)
        pred_dict["dete_pred"] = {"tot_pa": len(result[0]['rois'])}
        return pred_dict

    def __str__():
        return "dete_pred"

class DetCouPredProcedure:
    """Call this class to perform all prediction operations and return a
    dictionary in which there are elements as image path, detection preds,
    number of plagues, error and message if necessary 
    """

    @staticmethod
    def call_operations(**kwargs):
        predictions_dict = kwargs
        predictions_dict["radon_image_path"] = kwargs["input_img_path"]
        predictions_dict["error"] = 0
        predictions_dict["message"] = "No hay error"

        for operation in DetCouPredOperations.__subclasses__():
            predictions_dict = operation.operation(**predictions_dict)
            if predictions_dict['error'] != 0:
                break
        return predictions_dict

