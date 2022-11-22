"""
    tests to work with samples factory
"""
from IA_ModelsToolBox import samples_factory


BackgroundRefImgPath = r'/home/erickmfs/Pictures/background_ref.jpg'
BackgroundRefImgRoIPath = r'/home/erickmfs/Pictures/background_ref_roi.jpg'
#SampleUnitsPath = r'/home/erickmfs/plagas/test'
SampleUnitsPath = r'/home/erickmfs/plagas/2ndgen/1rst_samples'
#smpl_imgs_path = r'/home/erickmfs/plagas/synths1'
smpl_imgs_path = r'/home/erickmfs/plagas/2ndgen/sampleimg'
#smpl_img_masks_path = '/home/erickmfs/plagas/synthmasks1'
smpl_img_masks_path = '/home/erickmfs/plagas/2ndgen/samplemask'
max_smpl_units = 40
fsmplsnum = 8
SampleChoosingTimes = 50
imgsPath = r'/home/erickmfs/trampas/trampas_plagas_sinteticas' #reference sample image
roiimgsPath = r'/home/erickmfs/trampas/trampas_plagas_sinteticas_roi'
color_obj_type = "light" #or light for lighter objts
img_dim = (960, 720)#(600, 800)
dim_percentage = 0.4


"""
BackgroundRefImgPath = r'/home/erickmfs/Pictures/background_ref.jpg'
BackgroundRefImgRoIPath = r'/home/erickmfs/Pictures/background_ref_roi.jpg'
SampleUnitsPath = r'/home/erickmfs/RadonImages'
smpl_imgs_path = r'/home/erickmfs/Pictures/radon_synths/randon_imgs'
smpl_img_masks_path = r'/home/erickmfs/Pictures/radon_synths/radon_masks'
max_smpl_units = 40
fsmplsnum = 5
SampleChoosingTimes = 3 #320
imgsPath = r'/home/erickmfs/Pictures/randon_refimgs'
roiimgsPath = r'/home/erickmfs/Pictures/randon_refimgs'
color_obj_type = "dark" #or light for lighter objts
img_dim = (1280, 960)
dim_percentage = 1.0
"""

sf = samples_factory.SampleFactory(
    BackgroundRefImgPath,
    BackgroundRefImgRoIPath,
    SampleUnitsPath,
    fsmplsnum,
    SampleChoosingTimes,
    smpl_imgs_path,
    smpl_img_masks_path,
    max_smpl_units,
    imgsPath,
    roiimgsPath,
    color_obj_type,
    img_dim,
    dim_percentage
)

#Start factory of samples
sf.StartFactoryProcess()


