"""
    Testing PlagueDetCou
"""
import PlagueDetCou
from IA_ModelsToolBox import utils

print("Importing PlagueDetCou")

kwargs = {
    "aimodel_weights_path": r'/home/erickmfs/Downloads/unet_model/cp.ckpt',
    "input_img_path": r'/home/erickmfs/plagas/synths1/img_10.png',
    "plague_trshld": 50
}

predictions = PlagueDetCou.DetCouPredProcedure.call_operations(**kwargs)

print(f"Predictions: {predictions}")

utils.show_image("prediction results", predictions["plague_image"])

