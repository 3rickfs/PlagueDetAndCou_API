"""Utils for processing images
"""
import os
from abc import ABC, abstractmethod
from random import randint, seed, random
from base64 import b64encode
import time


import cv2
import numpy as np
import pygame

from IA_ModelsToolBox import BrownianMotion
from IA_ModelsToolBox import ScenarioCreation

WHITE = 255, 255, 255 #background color

def img64base_encoding(img):
    """
        Encoding image to be sent trhough POST request
    """

    retval, buffer = cv2.imencode('.png', img)
    base64_bytes = b64encode(buffer)
    base64_string = base64_bytes.decode('utf-8')

    return base64_string

def draw_contour(img, contours, contour_id, contour_color, contour_thickness):
    """
        Drawing contours on specific predicted images
    """

    cv2.drawContours(
        img,
        contours,
        contour_id,
        contour_color,
        contour_thickness
    )

def get_binary_img(img):
    """
        Binarize the image
    """

    gimg = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    img_mask = cv2.threshold(gimg, 128, 255, cv2.THRESH_OTSU)
    #show_image('mask', img_mask[1])

    return img_mask[1]


def get_contours_from_img(mimg):
    """
        Search for contours on the image and get centroids.
        Args:
            mimg: input mask image that have many blobs that could be the specific object to be counted.
        Return:
            contours: detected objs contour
            centroids: detected obj centroids
    """

    centroids = []
    contours, hierarchies = cv2.findContours(
        mimg,
        cv2.RETR_LIST,
        cv2.CHAIN_APPROX_SIMPLE
    )
    for contour in contours:
        M = cv2.moments(contour)
        if M['m00'] != 0: #to know if the obj is too small
            cw = int(M['m10']/M['m00'])
            ch = int(M['m01']/M['m00'])
            centroids.append((cw, ch))

    return contours, centroids

def resize_img(img_path, dim):
    """
        Resize and save the image in the same folder as the orginal one
        Args:
            img_path: image file path
            dim: tupe: (new width, new height)
        Return:
            new and resized image absolute path
    """
    img = cv2.imread(img_path)
    img = cv2.resize(img, dim, interpolation=cv2.INTER_AREA)
    nimg_path = img_path.split(".")[0]+"-r.png"
    cv2.imwrite(nimg_path, img)

    return nimg_path


def start_pygame_env(
    scenario_width,
    scenario_height,
    body_width,
    body_height,
    img_path
):
    """
        Run pygame and create the graphical environment to display objects
        interaction each other.
        Return:
            environment screen
            ref_img_obj: reference image to display according obj coordinates
    """
    pygame.init()
    size = scenario_width, scenario_height
    screen = pygame.display.set_mode(size)
    screen.fill(WHITE)
    nimg_path = resize_img(img_path, (body_width, body_height))
    ref_img_obj = pygame.image.load(nimg_path)

    return screen, ref_img_obj


def generate_obj_attr(
    name,
    imgs_path,
    scenario,
    scenario_width,
    scenario_height,
    simulation_time,
    distance_motion,
    n_step,
    body_width,
    body_height,
    reproduction_prohibition_time
):
    """
        Return a dict based on an object attribuits randomically generated
    """
    objattr = {}
    objattr["name"] = name
    #to work with png files only
    objattr["img_path"] = os.path.join(imgs_path, name + ".png")
    #seed(1)
    objattr["xpos"] = randint(0, scenario_width)
    objattr["ypos"] = randint(0, scenario_height)
    objattr["i_theta"] = randint(0, 360)
    objattr["simulation_time"] = simulation_time #seconds
    objattr["dt"] = simulation_time/n_step
    objattr["mu"] = random()
    objattr["sigma"] = random()
    objattr["distance_motion"] = distance_motion
    objattr["time_vector"] = np.linspace(0,simulation_time,num=n_step)
    objattr["body_width"] = body_width
    objattr["body_height"] = body_height
    objattr["scenario"] = scenario
    objattr["reproduction_prohibition_time"] = reproduction_prohibition_time

    return objattr

def scale_angle(ang):
    """
        set the angle between 0 and 360 degress
    """

    spins = ang/360
    dcmls = spins - int(spins)

    #0 -> 360
    #1 -> 0
    #m = 360 - 0 / 0 - 1 
    fang = -360*(dcmls - 1)

    return fang

def A_in_B(A, B):
    """
        all elements in set A are part of B
            Arguments:
                A, B: set A and B
            Return:
                boolean True or False
    """

    for i in A:
        if i not in B:
            return False

    return True

def A_intersects_B(A, B):
    """
        If just one element in set A is in B, A intercepts B
            Arguments:
                A, B: set A and B
            Return:
                boolean True or False
    """

    for i in A:
        if i in B:
            return True

    return False


def show_image(image_name, img):
    """Call this function to display an image with opencv """

    # Call opencv module to show the image
    cv2.imshow(image_name, img)
    #loop to wait for a key press
    cv2.waitKey(0)
    #Destroy all opencv windows
    cv2.destroyAllWindows()


def save_img(img_path, img):
    """
        Save an image in the corresponding folder
    """
    cv2.imwrite(img_path, img)


def read_img_same_dim(img_path):
    """
        Read image without resizing
    """

    return cv2.imread(img_path)


def read_img(img_path, dim):
    """
        Read image and resize it according to given dimensions
        args:
            img_path: image path
            dim: width and height tuple
    """

    img = cv2.imread(img_path)
    return cv2.resize(img, dim, interpolation=cv2.INTER_AREA)


def get_img_dim(img_path):
    """
        Return image dimensions
    """

    img = cv2.imread(img_path)
    return img.shape


class ConvexCombinationOps(ABC):
    """Image convex combination class constructor
    """

    @abstractmethod
    def operation(**kwargs):
        """Interface to child classes
        """
        pass


class MatchImages(ConvexCombinationOps):
    """Take 2 images and resize them so that those have pixel correspondence
    """

    def operation(**kwargs):
        img1w, img1h, _ = kwargs["img_list"][0].shape
        img2w, img2h, _ = kwargs["img_list"][1].shape
        avrg_width = int((img1w + img2w)/2)
        avrg_height = int((img1h + img2h)/2)
        nimg1 = cv2.resize(kwargs["img_list"][0], (avrg_width, avrg_height))
        nimg2 = cv2.resize(kwargs["img_list"][1], (avrg_width, avrg_height))
        kwargs["nimg_dict"] = {"nimg1": nimg1, "nimg2": nimg2}
        #print(f"nimg1 shape: {nimg1.shape}")
        #print(f"nimg2 shape: {nimg2.shape}")

        return kwargs


class ApplyConCom(ConvexCombinationOps):
    """Take 2 images and apply convex combination to get a crossfaded third
    image
    """

    def operation(**kwargs):
        """
        Arguments:
            img1, img2: images to be merged
            alfa: proportion for img1 to be considered
            beta: proportion for img2 to be considered
        output:
            mimg: combined image
        """
        alfa, beta = 0.5, 0.5
        img1 = kwargs["nimg_dict"]["nimg1"]
        img2 = kwargs["nimg_dict"]["nimg2"]
        print(f"data type of img1: {img1.dtype}")
        mimg = alfa * img1 + beta * img2
        mimg = mimg.astype('uint8')
        #print(mimg)
        #cv2.imshow("new image", mimg)
        #loop to wait for a key press
        #cv2.waitKey(0)
        #Destroy all opencv windows
        #cv2.destroyAllWindows()
        #show_image("new image", mimg)

        return mimg


class ConvexCombination:
    """A convex combination class to call all the ops to create a third image
    from two images as a crossfade effect
    """

    @staticmethod
    def call_operations(**kwargs):
        for operation in ConvexCombinationOps.__subclasses__():
            kwargs = operation.operation(**kwargs)

        return kwargs


class ImageUtils():
    """Class constructor for loading and processing images
    """

    def __init__(self):
        """Get atribuits when constructing the object
        """
        #self.imgpath = imgpath
        self.img_list = []

    def load_images(self, img_paths):
        """Load images
        Arguments:
            imgs_names: disctionary for images names

        """

        img_list = []
        for img_path in img_paths:
            #img_list.append(cv2.imread(os.path.join(self.imgpath,
            #                                        imgs_names[name])))
            img_list.append(cv2.imread(img_paths[img_path]))
            #print(img_list[-1])
            #displaying loaded image
            #cv2.imshow(name, img_list[-1])
            #loop to wait for a key press
            #cv2.waitKey(0)
            #Destroy all opencv windows
            #cv2.destroyAllWindows()
            #show_image(img_path, img_list[-1])

        self.img_list = img_list

    def MergeImages(self):
        """Merge 2 images using convex combination

        return:
            mimg: an image as a product of combining two images listed on
            img_list
        """

        mimg = ConvexCombination.call_operations(img_list = self.img_list)

        return mimg


class ImgObject():
    """
        Image object class to create an object that interact to others
        in order to combine them so that we can get synthetic images
    """

    def __init__(self, name, img_path, xpos,
                 ypos, i_theta, simulation_time,
                 dt, mu, sigma, distance_motion,
                 time_vector, body_width,
                 body_height, scenario,
                 reproduction_prohibition_time
                ):

        self.name = name
        self.img_path = img_path
        self.mergers_set = set([]) #to avoid repeted repros
        self.parents_set = set([]) #to avoid almost identical repros
        self.children_set = set([]) #the same as above
        self.reproduction_availability_delay = -1 #time step number
        self.reproduction_prohibition_time = reproduction_prohibition_time
        self.n_xpos = xpos #next x position
        self.n_ypos = ypos #next y position
        self.r_theta = i_theta #4 possile r_theta values: 0, 90, 180  and 270
        self.c_theta = self.r_theta #current theta to be changed by brownian.
        self.n_steps = int(simulation_time/dt)
        self.brownian_behaviour = BrownianMotion.Brownian(
            self.n_steps,
            time_vector,
            i_theta,
            mu,
            sigma
        )
        self.distance_motion = distance_motion
        self.c_bv = 0 #current brownian value
        self.c_scenario_knowledge = scenario.scenario_tuples
        self.c_scenario_centroid = scenario.centroid
        self.c_scenario_width = scenario.width
        self.c_scenario_height = scenario.height
        self.body_width = body_width
        self.body_height = body_height
        #self.positional_obj_centroids_correction()
        #self.body = self.get_obj_body_values()
        self.body = []
        self.get_according_body_values()
        self.c_xpos = self.n_xpos
        self.c_ypos = self.n_ypos
        scenario.add_an_img_obj(self) #scenario must to know how is in

    def reproduction_closing_process(self, objname, newobjname):
        """
            Close reproduction process running relevant control functions
            in order to update variables e.g. parents and children sets

            Args:
                objname: pair obj name that it is merging with this obj
                newobjname: result of mergin new obj name
        """

        self.mergers_set.add(objname)
        self.children_set.add(newobjname)
        self.set_reproduction_prohibition_time()


    def obj_reproduction_availability(self):
        """
            Return:
                True if obj can be merged with another one
                False the obj must wait until it has reproduction availability
        """

        if self.reproduction_availability_delay < 0:
            return True
        else:
            return False

    def set_reproduction_prohibition_time(self):
        """
            Return:
                updated self.reproduction_availability_delay
        """

        self.reproduction_availability_delay = self.reproduction_prohibition_time

    def reduce_reproduction_prohibition_time(self):
        """
            While reproduction avaialbility time is greater than 0 the obj
            will cannot merge with other then there is no reproduction effect
        """

        self.reproduction_availability_delay -= 1

    def get_obj_body_values(self):
        """
            A list of tuples coordinates that the body is filling in the
            2D space
        """

        hi = int(self.n_ypos - self.body_height/2)
        hf = int(self.n_ypos + self.body_height/2 + 1)
        wi = int(self.n_xpos - self.body_width/2)
        wf = int(self.n_xpos + self.body_width/2 + 1)
        tw = [n for n in range(wi, wf)]
        #if hi < 0: hc = -1*hi #height correction in case there are negatives
        #else: hc = 0
        #if wi < 0: wc = -1*wi #width correction
        #else: wc = 0
        body_list = []
        for h in range(hi, hf, 1):
            for w in tw:
                body_list.append((w, h))

        return body_list

    def positional_obj_centroids_correction(self):
        """
            Iterate new x and y coordinates until whole obj body inside the
            scenario
        """
        while(True):
            body_set = set(self.get_obj_body_values())
            scenario_set = set(self.c_scenario_knowledge)
            #body_into = True
            if not A_in_B(A=body_set, B=scenario_set):
            #for bt in body_set:
            #    if bt not in scenario_set: body_into = False
            #if not body_into:
                if self.n_xpos > self.c_scenario_centroid[0]:
                    self.n_xpos -= 4
                if self.n_xpos < self.c_scenario_centroid[0]:
                    self.n_xpos += 4
                if self.n_ypos > self.c_scenario_centroid[1]:
                    self.n_ypos -= 4
                if self.n_ypos < self.c_scenario_centroid[1]:
                    self.n_ypos += 4
            else:
                break

    def get_according_body_values(self):
        """
            Apply positional obj centroid correction and body values
            updating
        """

        self.positional_obj_centroids_correction()
        self.body = self.get_obj_body_values()

    def update_position(self, newxpos, newypos):
        self.c_xpos = newxpos
        self.c_ypos = newypos

    def get_x_position(self):
        return self.c_xpos

    def get_y_position(self):
        return self.c_ypos

    def get_img_path(self):
        return self.img_path

    def get_name(self):
        return self.name

    def set_scenario_knowledge(self, Scenario):
        self.c_scenario_knowledge = Scenario

    def get_scenario_knowledge(self):
        return self.c_scenario_knowledge

    def calculate_next_theta_value(self, i):
        self.theta = self.brownian_behaviour.geometric_brownian_motion(i)

    def calculate_next_x_position(self):
        self.n_xpos = self.c_xpos + self.distance_motion * np.cos(self.theta)

    def calculate_next_y_position(self):
        self.n_ypos = self.c_ypos + self.distance_motion * np.sin(self.theta)

    def print_obj_status(self):
        print("********************************************")
        print(f"Obj Name: {self.name}")
        print(f"image path: {self.img_path}")
        print("Scenario knowledge stablishment:",
              f"{True if self.c_scenario_knowledge!=None else False}")
        print(f"n_steps: {self.n_steps}")
        print(f"motion distance: {self.distance_motion}")
        print(f"current brownian value: {self.c_bv}")
        print(f"current x position: {self.c_xpos}")
        print(f"current y position: {self.c_ypos}")
        print(f"reference theta: {self.r_theta}")
        print(f"current theta: {self.c_theta}")
        print(f"current Brownian value: {self.c_bv}")
        print(f"body list: {self.body}")
        print(f"obj's list of mergers: {self.mergers_set}")
        print("********************************************")

class DreamingMachineOps(ABC):
    """
        Dreaming operations
    """

    @abstractmethod
    def operation(**kwargs):
        pass

class UpdateObjectScenarioKnowledge(DreamingMachineOps):
    """
        Update the object's knowledge about the scenario's environment.
        Staff like time, scenario changes etc.
    """

    def operation(**kwargs):
        print("Update object scenario knowledge")
        for obj in kwargs["objs_list"]:
            obj.set_scenario_knowledge(kwargs["scenario"].scenario_tuples)

        return kwargs

class UpdateObjectsPosition(DreamingMachineOps):
    """
    Update position to each object
        arguments:
            loo: list of objects

        Return:
            loo: an updated loo
    """

    def operation(**kwargs):
        print("Update Objects Position")
        for obj in kwargs["objs_list"]:
            bv = obj.brownian_behaviour.geometric_brownian_motion(
                kwargs["i_step"]
            )
            if obj.c_bv != 0:
                if obj.c_bv < bv:
                    r = bv - obj.c_bv
                    obj.c_bv = obj.c_bv + r
                else:
                    r = obj.c_bv - bv
                    obj.c_bv = obj.c_bv - r
                    #obj.c_bv = obj.c_bv - bv
                #if obj.c_bv < 0: obj.c_bv = 0
            else:
                obj.c_bv = bv
            obj.c_theta = obj.r_theta + obj.c_bv
            obj.r_theta = obj.c_theta
            if obj.c_theta > 360: obj.c_theta -= obj.c_bv
            obj.n_xpos = obj.n_xpos + obj.distance_motion*np.cos(
                obj.c_theta*np.pi/180
            )
            obj.n_ypos = obj.n_ypos + obj.distance_motion*np.sin(
                obj.c_theta*np.pi/180
            )

        return kwargs

class RefreshBodyAwareness(DreamingMachineOps):
    """
        Update body values according to obj's centroid
            arguments:
                loo: list of objects
            return:
                loo: with updated body list values
    """

    def operation(**kwargs):
        print("Refresh body awareness")
        for obj in kwargs["objs_list"]:
            obj.body = obj.get_obj_body_values()

        return kwargs

class ObjectBodyPositionCorrection(DreamingMachineOps):
    """
        Looking for collitions between the objt and scenario
    """

    def operation(**kwargs):
        print("Update Object body status")
        for obj in kwargs["objs_list"]:
            body_set = set(obj.body)
            scenario_set = set(obj.c_scenario_knowledge)
            #Updating current x and y coordinates
            if A_in_B(A=body_set, B=scenario_set):
                obj.c_xpos = obj.n_xpos
                obj.c_ypos = obj.n_ypos
            else:
                body_right_edge = obj.n_xpos + (obj.body_width/2)
                body_up_edge = obj.n_ypos + (obj.body_height/2)
                if body_right_edge > obj.c_scenario_width \
                    and body_up_edge < obj.c_scenario_height:
                    #then obj has overpassed scenario's right edge
                    #change the obj direction to 
                    obj.r_theta = 180 #turn left

                if body_right_edge < obj.c_scenario_width \
                    and body_up_edge > obj.c_scenario_height:
                    #then obj has overpassed scenario's up edge
                    obj.r_theta = 270 #turn down

                #as the scenario is located at the 2D cartesian space origin
                body_left_edge = obj.n_xpos - (obj.body_width/2)
                body_bottom_edge = obj.n_ypos - (obj.body_height/2)
                if body_left_edge < 0 and body_bottom_edge > 0:
                    #then obj has overpassed scenario's left edge
                    obj.r_theta = 0 #turn right

                if body_left_edge > 0 and body_bottom_edge < 0:
                    #then obj has overpassed scenario's bottom edge
                    obj.r_theta = 90 #turn up

            #to have no contact between the body and scenario
            obj.get_according_body_values()

        return kwargs

class GenerateAnimation(DreamingMachineOps):
    """
        Generate an animation to see how image objects move and interact
        in a 2D universe
    """

    def operation(**kwargs):
        if kwargs["graphical_env"]:
            print("Generate animation")
            print(f"pygame: {pygame}")
            kwargs["screen"].fill(WHITE)
            for obj in kwargs["objs_list"]:
                kwargs["screen"].blit(kwargs["rimg"], (obj.c_xpos, obj.c_ypos))

            pygame.display.flip()

        return kwargs


class IntersectionsSearch(DreamingMachineOps):
    """
        This class is to seek if there exist any collintion of the object
        subspace, that is the object "body touching others"

        Return:
            list of sets of objects that collide each other
    """

    def operation(**kwargs):
        print("Intersection search")
        n = 0
        objslist = kwargs["objs_list"].copy()
        iobjs = {}
        for i, objA in enumerate(objslist):
            _ = objslist.pop(i)
            for objB in objslist:
                objA_body_set = set(objA.body)
                objB_body_set = set(objB.body)
                if A_intersects_B(A=objA_body_set, B=objB_body_set): # and objA != objB:
                    #there is an intersection between two different objects!
                    iobjs["intrcptn_"+str(n)] = dict(
                        zip(
                            ["objA", "objB"],
                            [objA, objB]
                        )
                    )
                    n += 1
                    #Body direction change
                    objA.r_theta = scale_angle(objA.r_theta + 180)
                    objB.r_theta = scale_angle(objB.r_theta + 180)

        kwargs["intrcptd_objs"] = iobjs

        return kwargs

class IntersectionsNumberProhibition(DreamingMachineOps):
    """
        Taking care of processing saturation avoiding mergings
        according to the numbwer of intersections
    """

    def operation(**kwargs):
        print("Intersections number prohibition")
        intrs_num = len(kwargs["intrcptd_objs"])
        if intrs_num > kwargs["max_intrcptd_objs"]:
            nio = {}
            n = 0
            for i, io in enumerate(kwargs["intrcptd_objs"]):
                if i < kwargs["max_intrcptd_objs"]:
                    #new intersected objs 
                    nio["intrcptn_"+str(n)] = dict(
                        zip(
                            ["objA", "objB"],
                            [
                                kwargs["intrcptd_objs"][io]["objA"],
                                kwargs["intrcptd_objs"][io]["objB"]
                            ]
                        )
                    )
                    n += 1

            kwargs["intrcptd_objs"] = nio

        return kwargs


class ApplyObjectCombination(DreamingMachineOps):
    """
        This class reads the list of collided objects sets and perform a
        convex combination of two or more  objects

        Return:
            list of new images as a product of different sets of objects that
            got mixed
    """

    def operation(**kwargs):
        print("Apply object combination")
        mimgs_dict = {}
        for i, io in enumerate(kwargs["intrcptd_objs"]):
            #print(f"objA: {kwargs['intrcptd_objs'][io]['objA']}")
            objA = kwargs["intrcptd_objs"][io]["objA"]
            objB = kwargs["intrcptd_objs"][io]["objB"]
            #Rules of reproductions between objs
            if not objA.name in objB.mergers_set and \
               (not objA.name in objB.children_set or \
                not objB.name in objA.parents_set) and \
               (objA.obj_reproduction_availability and \
                objB.obj_reproduction_availability):

                iu = ImageUtils()
                print(f"imgpathA: {objA.img_path}")
                iu.load_images({"imgpath1": objA.img_path,
                                "imgpath2": objB.img_path})
                mimg = iu.MergeImages()
                #show_image("Merged image", mimg) 

                #Mergings dictionary
                mimgs_dict["mimg"+str(i)] = dict(
                    zip(
                        ["name", "img", "parent1", "parent2"],
                        ["obj"+str(kwargs["scenario"].objs_number+i),
                          mimg,
                          objA.name,
                          objB.name
                        ]
                    )
                )

                #Close the reproduction updating relevant variables
                objA.reproduction_closing_process(
                    objB.name,
                    mimgs_dict["mimg"+str(i)]["name"]
                )
                objB.reproduction_closing_process(
                    objA.name,
                    mimgs_dict["mimg"+str(i)]["name"]
                )

        kwargs["new_mimgs"] = mimgs_dict

        return kwargs

class UpdateObjectProhibitionTimeDelay(DreamingMachineOps):
    """
        Reduce the time step needed to know if the obj is able of being merged
    """

    def operation(**kwargs):
        print("Update obj prohibition time delay")
        for obj in kwargs["objs_list"]:
            obj.reduce_reproduction_prohibition_time()

        return kwargs

class CreateNewImgObject(DreamingMachineOps):
    """
        Class to create new objects from a list of new images after have been
        combined each other
    """

    def operation(**kwargs):
        print("Create new image object")
        for ni in kwargs["new_mimgs"]:
            save_img(
                os.path.join(
                    kwargs["img_path_env"],
                    kwargs["new_mimgs"][ni]["name"]+".png"
                ),
                kwargs["new_mimgs"][ni]["img"]
            )

            ioa = generate_obj_attr(
                kwargs["new_mimgs"][ni]["name"],
                kwargs["img_path_env"],
                kwargs["scenario"],
                kwargs["scenario"].width,
                kwargs["scenario"].height,
                kwargs["simulation_time"],
                kwargs["distance_motion"],
                kwargs["n_step"],
                kwargs["body_width"],
                kwargs["body_height"],
                kwargs["reproduction_prohibition_time"]
            )

            nobj = ImgObject(**ioa)
            #adding parent set value
            nobj.parents_set.add(kwargs["new_mimgs"][ni]["parent1"])
            nobj.parents_set.add(kwargs["new_mimgs"][ni]["parent2"])
            
            kwargs["scenario"].add_an_img_obj(nobj)
            kwargs["objs_list"].append(nobj)

        return kwargs


class ProduceSpaceTime(DreamingMachineOps):
    """
        Update time variables and produce a pause
    """

    def operation(**kwargs):
        kwargs["i_step"] += 1 #kwargs["n_step"]
        time.sleep(kwargs["dt"])

        return kwargs

class PrintCurrentSimulationStatus(DreamingMachineOps):
    """
        Print current obj's and simulation's status
    """

    def operation(**kwargs):
        if kwargs["verbose"]:
            print("-------------------------------------------------------------")
            print(f"i step: {kwargs['i_step']}")
            print(f"intersections: {kwargs['intrcptd_objs']}")
            print(f"objs_list: {kwargs['objs_list']}")
            print(f"Simulation time: {kwargs['time_vector'][kwargs['i_step']-1]}")
            print("-------------------------------------------------------------")
            print("IMAGE OBJECTS:")
            for obj in kwargs["objs_list"]:
                obj.print_obj_status()

        return kwargs

class DreamingMachine:
    """
        This class makes it possible to instanciate an object to create image
        experiences in a simulation of interacting image objects that got
        combined when collide each other
    """

    #img_path_env = None

    @staticmethod
    def start_dreaming(**kwargs):
        print(f"kwargs[objs_list]: {kwargs['objs_list']}")
        while(kwargs["n_step"] > kwargs["i_step"]):
            for operation in DreamingMachineOps.__subclasses__():
                kwargs = operation.operation(**kwargs)

        return kwargs

#update position to each object
#generate graphics if required
#looking for intersections between objects
#apply combination (generate another object) if objects collide

#orientation applied when producing set of objects images


if __name__ == "__main__":
    #ipath = r"/home/erickmfs/Pictures/"
    iu = ImageUtils() #ipath)
    iu.load_images({"imgpath1": r"/home/erickmfs/Pictures/i1.png", "imgpath2": r"/home/erickmfs/Pictures/i5.png"})
    res = iu.MergeImages()
    #DreamingMachine.start_dreaming(nothing="nothing")

    #print(res)








