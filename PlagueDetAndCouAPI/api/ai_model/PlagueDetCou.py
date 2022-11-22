"""Module to provide acess to the plague detection and counting model
"""

import sys
import os
from abc import ABC, abstractmethod

import numpy as np

from IA_ModelsToolBox import utils, IAmodels

class DetCouPredOperations():
    """Build an interface to prediction operation classes
    """

    @abstractmethod
    def operation(**kwargs):
        pass


class get_plague_image(DetCouPredOperations):
    """Return the plague image"""

    def operation(**kwargs):
        print("Getting the corresponding plague image")
        #plague image original dimensions
        kwargs["pimg_odim"] = utils.get_img_dim(kwargs["input_img_path"])
        dim = (128, 128)
        img = utils.read_img(kwargs["input_img_path"], dim)
        img = np.expand_dims(img, axis=0)
        kwargs["plague_image"] = img
        #pred_dict["error"] = 1
        #pred_dict["message"] = "Error en get_radon_img"

        return kwargs

    def __str__():
        return "radon_img"


class build_model(DetCouPredOperations):
    """
        Build the corresponding model, in this case Unet based on ResNet18
    """

    def operation(**kwargs):
        print("Building the model")
        kwargs["aimodel"] = IAmodels.get_unet_ResNet18()
        return kwargs


class load_weights(DetCouPredOperations):
    """
        Load model's weights
    """

    def operation(**kwargs):
        print("Loading weights to the model")
        kwargs["aimodel"].load_weights(kwargs["aimodel_weights_path"])

        return kwargs


class predict_mask(DetCouPredOperations):
    """Make a prediction of plague image mask
    """

    def operation(**kwargs):
        print("Making mask prediction")
        pmask = kwargs["aimodel"].predict(kwargs["plague_image"])
        pmask = pmask[0] * 255
        pmask = pmask.astype(np.uint8)
        kwargs["predicted_mask"] = pmask

        return kwargs


class binarize_predicted_mask(DetCouPredOperations):
    """
        Transform the RGB predicted image into a binary one
    """

    def operation(**kwargs):
        print("Binarizing predicted mask image")
        print(f"pmask shape: {kwargs['predicted_mask'].shape}")
        kwargs["bi_pmask"] = utils.get_binary_img(kwargs["predicted_mask"])

        return kwargs


class get_detected_obj_contours(DetCouPredOperations):
    """
        Get contours and their corresponding centroids using utils AI toolbox

    """

    def operation(**kwargs):
        print("Getting contours from detected plagues")
        contours, centroids = utils.get_contours_from_img(
            kwargs["bi_pmask"]
        )
        print(f"Number of contours: {len(contours)}")
        kwargs["plagues_contours"] = contours
        kwargs["plagues_centroids"] = centroids

        return kwargs

class count_detected_plagues(DetCouPredOperations):
    """
        Count corresponding plagues detected by the ai model
        There are two kinds of plagues:
            pa - Elasmopalpus Lignosellus: smaller fly
            pb - Spodoptera Frugiperda: bigger fly
    """

    def operation(**kwargs):
        print("Couting detected plagues")
        plague_trshld = kwargs["plague_trshld"]
        plagues = []
        tot_pa, tot_pb = 0, 0
        for contour in kwargs["plagues_contours"]:
            if plague_trshld > len(contour):
                #then it is pa
                plagues.append("pa")
                tot_pa += 1
            else:
                #then it is pb
                plagues.append("pb")
                tot_pb += 1

        kwargs["tot_pa"] = tot_pa
        kwargs["tot_pb"] = tot_pb
        kwargs["plagues"] = plagues

        return kwargs

    def __str__():
        return "coun_pred"


class get_prediction_img(DetCouPredOperations):
    """
        Create an image to represent prediction results labelling detected
        plagues according to the plague
    """

    def operation(**kwargs):
        print("Getting prediction image")
        #contour_id = -1 #all contours
        contour_thickness = 1

        for contour_id in range(len(kwargs["plagues_contours"])):
            if kwargs["plagues"][contour_id] == "pa":
                contour_color = (0, 0, 255)
            else: #then it's pb
                contour_color = (255, 0, 0)

            utils.draw_contour(
                kwargs["plague_image"][0],
                kwargs["plagues_contours"],
                contour_id,
                contour_color,
                contour_thickness
            )

        return kwargs


class DecodeImgFile(DetCouPredOperations):
    """
        In order to send the image to the cliente it is needed to get it
        as a base64 format.
    """

    def operation(**kwargs):
        print("Decode image file")
        #retval, buffer = cv2.imencode('.png', kwargs["plague_image"][0])
        #jbase64_byte = b64encode(buffer)
        base64_string = utils.img64base_encoding(kwargs["plague_image"][0])
        kwargs["base64_plague_img"] = base64_string

        return kwargs


class DetCouPredProcedure:
    """Call this class to perform all prediction operations and return a
    dictionary in which there are elements as image path, detection preds,
    number of radon marks, error and message if necessary
    """

    @staticmethod
    def call_operations(**kwargs):
        #kwargs = {}
        #kwargs["radon_image_path"] = kwargs["radon_image_path"]
        #kwargs["error"] = 0
        #kwargs["message"] = "No hay error"

        for operation in DetCouPredOperations.__subclasses__():
            kwargs = operation.operation(**kwargs)

        return {
            "base64_img_string": kwargs["base64_plague_img"],
            "tot_pa": kwargs["tot_pa"],
            "tot_pb": kwargs["tot_pb"]
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


