"""
Factory to produce samples so that a data set can be built
in order to train a U-Net model performs semantic segmentation
"""

import os
from abc import ABC, abstractmethod
from random import randint
import fnmatch
import numpy as np

import cv2

from . utils import show_image, save_img, read_img

def get_image_path(path, imgnum):
    """
        Get the image path from a specific image number
    """
    return os.path.join(
        path,
        os.listdir(path)[imgnum]
    )

def get_files_number(path):
    """
        Number of files in a folder
    """

    return len(fnmatch.filter(
                os.listdir(path), '*.*'
    ))

def find_out_roi(roi_set, smpl_img, img):
    """
        Find out the limits of the roi into the sample image
    """

    width, height = smpl_img.shape[:2]
    img_width, img_height = img.shape[:2]
    img_hlfwidth, img_hlfheight = img_width//2, img_height//2
    wmin, wmax, hmin, hmax = 0, 0, 0, 0
    while True:
        w, h = randint(0, width), randint(0, height)
        if (w, h) in roi_set:
            if (w-img_hlfwidth, h) in roi_set:
                if (w+img_hlfwidth, h) in roi_set:
                    if (w, h-img_hlfheight) in roi_set:
                        if (w, h+img_hlfheight) in roi_set:
                            wmin = w - img_hlfwidth
                            wmax = w + img_hlfwidth
                            hmin = h - img_hlfheight
                            hmax = h + img_hlfheight
                            break

    wmax += img_width - (wmax-wmin)
    hmax += img_height - (hmax-hmin)

    return wmin, wmax, hmin, hmax


def get_adjusted_img(mimg, img):
    """
        Resize the images according to the object mask shape
        args:
            mimg: image mask
            img: image
    """

    def pixel_detection():
        pide = False
        if len(mimg.shape)>2:
            if mimg[w,h,0] == 255 and \
            mimg[w,h,1] == 255 and \
            mimg[w,h,2] == 255:
                pide = True
        else:
            if mimg[w,h] > 0: pide = True
        return pide

    width, height = mimg.shape[:2]
    flg = False
    for h in range(height):
        for w in range(width):
            if pixel_detection():
                flg = True
                break
        if flg: break
    hmin = h
    #print(f"hmin: {hmin}")

    flg = False
    for h in range(height-1, -1, -1):
        for w in range(width):
            if pixel_detection():
                flg = True
                break
        if flg: break
    hmax = h
    #print(f"hmax: {hmax}")

    flg = False
    for w in range(width):
        for h in range(height):
            if pixel_detection():
                flg = True
                break
        if flg: break
    wmin = w
    #print(f"wmin: {wmin}")

    flg = False
    for w in range(width-1, -1, -1):
        for h in range(height):
            if pixel_detection():
                flg = True
                break
        if flg: break
    wmax = w
    #print(f"wmax: {wmax}")

    rez_mimg = mimg[wmin:wmax, hmin:hmax]
    rez_img = img[wmin:wmax, hmin:hmax, :]

    return rez_img, rez_mimg

def get_img_mask(img, objttype):
    """
        Generates the mask of the image
    """

    gimg = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    img_mask = cv2.threshold(gimg, 128, 255, cv2.THRESH_OTSU)[1]
    #show_image('mask', img_mask[1])
    if objttype == "dark":
        #print(f"mimg: {mimg}")
        img_mask = cv2.bitwise_not(img_mask)

    return img_mask

def rotate_image(mat, angle):
    """
    Rotates an image (angle in degrees) and expands image to avoid cropping
    """

    height, width = mat.shape[:2] # image shape has 3 dimensions
    image_center = (width/2, height/2) # getRotationMatrix2D needs coordinates in reverse order (width, height) compared to shape

    rotation_mat = cv2.getRotationMatrix2D(image_center, angle, 1.)

    # rotation calculates the cos and sin, taking absolutes of those.
    abs_cos = abs(rotation_mat[0,0])
    abs_sin = abs(rotation_mat[0,1])

    # find the new width and height bounds
    bound_w = int(height * abs_sin + width * abs_cos)
    bound_h = int(height * abs_cos + width * abs_sin)

    # subtract old image center (bringing image back to origo) and adding the new image center coordinates
    rotation_mat[0, 2] += bound_w/2 - image_center[0]
    rotation_mat[1, 2] += bound_h/2 - image_center[1]

    # rotate image with the new bounds and translated rotation matrix
    rotated_mat = cv2.warpAffine(mat, rotation_mat, (bound_w, bound_h))

    return rotated_mat

def get_imgn(selected_imgs, tot_imgs_num):
    """
        Get an image number that it was not choosen before
        args:
            selected_imgs: set of selected images
            tot_imgs_num: number of total images in samples folder
        return
            imgn: number of image
    """
    imgn = 0
    while(True):
        imgn = randint(0, tot_imgs_num)
        if not imgn in selected_imgs:
            selected_imgs.add(imgn)
            break

    return  imgn

class SamplesFactoryOps(ABC):
    """
        Factory of samples operations
    """

    @abstractmethod
    def operation(**kwargs):
        pass

class LoadRefImgs(SamplesFactoryOps):
    """
        Load corresponding reference images.
        - Background image
        - Background image RoI (Region of Interest)
    """

    def operation(**kwargs):
        print("Loading reference images")
        imgfilesnum = get_files_number(kwargs["imgsPath"])
        imgnum = randint(0,imgfilesnum-1)
        imgpath = get_image_path(kwargs["imgsPath"], imgnum)
        roiimgpath = get_image_path(kwargs["roiimgsPath"], imgnum)
        dim = kwargs["img_dim"] #(600, 800)
        bkgrefimg = read_img(imgpath, dim)
        #show_image("bkgrefimg", bkgrefimg)
        bkgrefimgroi = read_img(roiimgpath, dim)
        #show_image("bkgrefimgroi", bkgrefimgroi)
        kwargs["bkgrefimg"] = bkgrefimg
        kwargs["bkgrefimgroi"] = bkgrefimgroi

        return kwargs


class SampleROIsSearching(SamplesFactoryOps):
    """
        Images must have a small size due to resources limitations
        when training the IA models, this Op will get a ROI based
        on the sample masks
    """

    def operation(**kwargs):
        print("Sample ROI searching Operation")
        #for img, mimg in zip(kwargs["bkgrefimg"], kwargs["bkgrefimgroi"]):
        #for i in range(len(kwargs["bkgrefimgroi"])):
        img, mimg = kwargs["bkgrefimg"], kwargs["bkgrefimgroi"]
        #show_image('img',mimg)
        nimg, nmimg = get_adjusted_img(mimg, img)
        #show_image('img',nmimg)

        kwargs["bkgrefimg"], kwargs["bkgrefimgroi"] = nimg, nmimg

        return kwargs


class GetRoISets(SamplesFactoryOps):
    """
        Get a set containing all the pixels as coordinates from the GetRoISets
    """

    def operation(**kwargs):
        print("Getting sets of RoI coordinates")
        w, h, _ = kwargs["bkgrefimgroi"].shape
        roi_set = set()
        for i in range(w):
            for j in range(h):
                if kwargs["bkgrefimgroi"][i, j, 0] == 255:
                    #then it is a white pixel
                    roi_set.add((i, j))

        kwargs["roi_set"] = roi_set

        return kwargs


class LoadSampleUnits(SamplesFactoryOps):
    """
        Load corresponding sample units, that is object images
    """

    def operation(**kwargs):
        print("Loading sample units")
        SampleUnits = []
        dper = kwargs["dim_percentage"]
        for img_name in os.listdir(kwargs["SampleUnitsPath"]):
            img = cv2.imread(os.path.join(kwargs["SampleUnitsPath"], img_name))
            dim = (int(img.shape[:2][0]*dper), int(img.shape[:2][1]*dper))
            img = cv2.resize(img, dim, interpolation=cv2.INTER_AREA)
            SampleUnits.append(img)
        kwargs["SampleUnits"] = SampleUnits

        return kwargs


class GenerateSampleUnitsMask(SamplesFactoryOps):
    """
        Generate masks related to each sample unit
    """

    def operation(**kwargs):
        print("Generating sample units masks")
        SampleUnitMasks = []
        for img in kwargs["SampleUnits"]:
            mimg = get_img_mask(img, kwargs["color_obj_type"])
            #There is dark objts that need to be addressed

            SampleUnitMasks.append(mimg)

        kwargs["SampleUnitMasks"] = SampleUnitMasks

        return kwargs


class ProduceFinalSamples(SamplesFactoryOps):
    """
        Produce final samples to be part of the final dataset
    """

    def operation(**kwargs):
        print("Producing final samples")
        imgsnum = get_files_number(kwargs["SampleUnitsPath"])
        print(f"Number of Samplesunits: {imgsnum}")
        selected_imgs = set()
        smpl_imgs = []
        #Create the sample image mask list
        smpl_img_masks = []

        for s in range(kwargs["fsmplsnum"]):
            #Sample image creation
            smpl_img = kwargs["bkgrefimg"].copy()
            onum = randint(2, kwargs["max_smpl_units"])
            #Create the sample image mask 
            smpl_img_mask = np.zeros(smpl_img.shape[:2], dtype="uint8")
            for o in range(onum):
                #Select image
                #inum = get_imgn(selected_imgs, imgsnum) #to select images that were not choosen before
                inum = randint(0, imgsnum-1)
                img = kwargs["SampleUnits"][inum]
                mask = kwargs["SampleUnitMasks"][inum]
                #Mask image centroid
                contours, hierarchies = cv2.findContours(
                    mask,
                    cv2.RETR_LIST,
                    cv2.CHAIN_APPROX_SIMPLE
                )
                #It is known that it'll have only one contour
                #but for sure
                cw, ch = 0, 0
                #print(contours)
                #print(len(contours[0]))
                cl = [len(contours[i]) for i in range(len(contours))]
                clmax = max(cl)
                max_cou_i = 0
                for i in range(len(cl)):
                    if cl[i] == clmax: max_cou_i = i
                M = cv2.moments(contours[max_cou_i])
                if M['m00'] != 0: #avoiding divisions by zero
                    cw = int(M['m10']/M['m00']) #x-centroid along the width of the image
                    ch = int(M['m01']/M['m00']) #y-centroid along the height of the image
                else:
                    raise Exception("There was a zero-division while getting centroids!")

                #Having the centroids we need that it's up to apply masking to img
                masked = cv2.bitwise_and(img, img, mask=mask)

                #maybe here it is possible to include rotations
                rot_ang = randint(0, 360) #random rotation angle
                rotated_img = rotate_image(masked, rot_ang)
                #Get the new mask
                rotated_img_mask = get_img_mask(
                    rotated_img,
                    "light" #kwargs["color_obj_type"]
                )

                #rezise image according to obj shape
                adjusted_rotated_img, adjusted_rotated_img_mask = get_adjusted_img(
                    rotated_img_mask,
                    rotated_img
                )
                inv_adj_rot_img_mask = cv2.bitwise_not(adjusted_rotated_img_mask)
                #show_image("inv", inv_adj_rot_img_mask)

                #Find out the roi in the sample img
                wmin, wmax, hmin, hmax = find_out_roi(
                    kwargs["roi_set"], smpl_img, adjusted_rotated_img
                )
                roi = smpl_img[wmin:wmax, hmin:hmax]
                #Getting the adjusted rotated image mask into the bigger sample image mask
                smpl_img_mask[wmin:wmax, hmin:hmax] = adjusted_rotated_img_mask
                #Black-out the area of the sample image in ROI
                img_bo = cv2.bitwise_and(roi, roi, mask= inv_adj_rot_img_mask)
                #Put obj in ROI and modify the sample image
                mod_roi = cv2.add(img_bo, adjusted_rotated_img)
                smpl_img[wmin:wmax, hmin:hmax, :] = mod_roi

            #add the sample image to the list
            smpl_imgs.append(smpl_img)
            smpl_img_masks.append(smpl_img_mask)
            #show_image("New sample", smpl_img)

        #New staff
        kwargs["smpl_imgs"] = smpl_imgs
        kwargs["smpl_img_masks"] = smpl_img_masks

        return kwargs


class SaveSmplImgs(SamplesFactoryOps):
    """
        Save the all new sample images generated including objts
    """

    def operation(**kwargs):
        print("Saving sample images")
        imgsnum = get_files_number(kwargs["smpl_imgs_path"])
        index = range(len(kwargs["smpl_imgs"]))
        for (i, img, imgmask) in zip(
            index,
            kwargs["smpl_imgs"],
            kwargs["smpl_img_masks"]
        ):
            ipath = os.path.join(
                kwargs["smpl_imgs_path"],
                "img_"+str(imgsnum+i)+".png"
            )
            impath = os.path.join(
                kwargs["smpl_img_masks_path"],
                "img_"+str(imgsnum+i)+".png"
            )
            save_img(ipath, img)
            save_img(impath, imgmask)

        return kwargs


class ProduceSamples:

    @staticmethod
    def start_producing_samples(**kwargs):
        print("The process of producing samples has started")
        for s in range(kwargs["SampleChoosingTimes"]):
            for operation in SamplesFactoryOps.__subclasses__():
                kwargs = operation.operation(**kwargs)

        return kwargs


class SampleFactory():
    """
        Call this class to get access to the synthetizing engine of image samples based on unit object images
    """

    def __init__(
        self,
        bckgdRefImgPath,
        bckgdRefImgRoIPath,
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
    ):
        self.bckgdRefImgPath = bckgdRefImgPath
        self.bckgdRefImgRoIPath = bckgdRefImgRoIPath
        self.SampleUnitsPath = SampleUnitsPath
        self.fsmplsnum = fsmplsnum
        self.smpl_imgs_path = smpl_imgs_path
        self.smpl_img_masks_path = smpl_img_masks_path
        self.max_smpl_units = max_smpl_units
        self.imgsPath = imgsPath
        self.roiimgsPath = roiimgsPath
        self.SampleChoosingTimes = SampleChoosingTimes
        self.color_obj_type = color_obj_type
        self.img_dim = img_dim
        self.dim_percentage = dim_percentage

    def StartFactoryProcess(self):
        ProduceSamples.start_producing_samples(
            bckgdRefImgPath=self.bckgdRefImgPath,
            bckgdRefImgRoIPath=self.bckgdRefImgRoIPath,
            SampleUnitsPath=self.SampleUnitsPath,
            fsmplsnum=self.fsmplsnum,
            SampleChoosingTimes=self.SampleChoosingTimes,
            smpl_imgs_path=self.smpl_imgs_path,
            smpl_img_masks_path=self.smpl_img_masks_path,
            max_smpl_units=self.max_smpl_units,
            imgsPath=self.imgsPath,
            roiimgsPath=self.roiimgsPath,
            color_obj_type = self.color_obj_type,
            img_dim = self.img_dim,
            dim_percentage = self.dim_percentage
        )












