import os
from json import dumps

import requests
import pandas as pandas
from PIL import Image
from django.shortcuts import render
from django.conf import settings
from django.http import JsonResponse
from rest_framework.response import Response
from rest_framework.decorators import api_view

#from api.IA_ModelsToolBox import RadonDetCou
from IA_ModelsToolBox import utils
from .ai_model import PlagueDetCou

# Create your views here.
@api_view(['POST'])
def get_predictions(request):
    """Call this method to get predictions from plagues images and return
    a dictionary containing detection and counting preds as well as
    an error and a message if necessary
    """

    try:
        print("Loading image")
        url = request.POST.get('url')
        head, tail = os.path.split(url)
        r = requests.get(url, allow_redirects=True)
        imgpath = os.path.join(settings.MEDIA_ROOT, tail)
        open(imgpath, 'wb').write(r.content)
        print(f"Image loaded into: {imgpath}")
        img_frmt = imgpath.split('.')[1]
        if (img_frmt.lower() == "jpeg" or
            img_frmt.lower() == "jpg" or
            img_frmt.lower() == "png"):
            awp = os.path.join(settings.MODELS, "cp.ckpt")
            kwargs = {
                "aimodel_weights_path": awp,
                "input_img_path": imgpath,
                "plague_trshld": 50,
                "pimgfolderpath": settings.MEDIA_ROOT
            }
            predictions = PlagueDetCou.DetCouPredProcedure.call_operations(
                **kwargs
            )
            #predictions["64base_img_string"] = img64base_encoding(imgpath)
            predictions["error"] = 0
            predictions["message"] = "Success"
        else:
            predictions["error"] = 1
            predictions["message"] = """
                Image file have no correct format: jpeg, jpg, png
            """

    except Exception as e:
        predictions = {
            'error': 2,
            'message': str(e)
        }
print("Sending data back to the client")
    predictions_json = dumps(predictions)
return JsonResponse(predictions_json, safe=False)
