from IA_ModelsToolBox import RadonDetCou
kwargs = {
                "aimodel_weights_path": 'mask_rcnn_polilla_0017.h5',
                "input_img_path": 'C:/Users/prel1/Downloads/imagenes/images/imagen64.jpg',
                "plague_trshld": 50,
                "pimgfolderpath": ''
            }
predictions = RadonDetCou.DetCouPredProcedure.call_operations(**kwargs)