"""
Dreaming machine simulation test module
"""
import os
import numpy as np

from IA_ModelsToolBox import ScenarioCreation, utils

class SimulationTest:

    @staticmethod
    def test_objectknowledge_setting():
        n_step = 200
        total_time = 10 #seconds
        time_vector = np.linspace(0,total_time,num=n_step)
        dt = total_time/n_step
        s0 = 2
        mu = 0.2
        sigma = 0.68
        xpos = 0
        ypos = 0
        i_theta = 5
        obj_width = 20
        obj_height = 20
        distance_motion = 7
        time_vector = np.linspace(0, total_time, num=n_step)
        scenario_width = 200
        scenario_height = 200
        scenario = ScenarioCreation.BasicScenario(scenario_width, scenario_height, time_vector)
        scenario.create()
        imgs_path = r"/home/erickmfs/plagas/1"
        #img_path_obj1 = r"/home/erickmfs/Pictures/i1.png"
        #img_path_obj2 = r"/home/erickmfs/Pictures/i2.png"
        #img_path_obj3 = r"/home/erickmfs/Pictures/i4.png"
        #rimg_path = r"/home/erickmfs/Pictures/i2-r.png"
        #rimg_path = r"/home/erickmfs/Pictures/u2.jpg"
        rimg_path = r"/home/erickmfs/Pictures/i1.png"
        reproduction_prohibition_time = 30
        max_intrcptd_objs = 20

        #obj1 = utils.ImgObject("Obj1",img_path_obj1,
        #                       xpos, ypos, i_theta, total_time,
        #                       dt, mu, sigma, distance_motion, time_vector,
        #                       obj_width, obj_height, scenario,
        #                       reproduction_prohibition_time,
        #                      )
        #obj2 = utils.ImgObject("Obj2", img_path_obj2,
        #                       xpos, ypos, i_theta, total_time,
        #                       dt, mu, sigma, distance_motion, time_vector,
        #                       obj_width, obj_height, scenario,
        #                       reproduction_prohibition_time
        #                      )

        #obj3 = utils.ImgObject("Obj3",img_path_obj3,
        #                       100, 100, i_theta, total_time,
        #                       dt, mu, sigma, distance_motion, time_vector,
        #                       obj_width, obj_height, scenario,
        #                       reproduction_prohibition_time
        #                      )
        #onames = ["rObj3", "rObj4", "rObj5", "rObj6"]
        onames = []
        for i, imgname in enumerate(os.listdir(imgs_path)):
            #onames.append("m"+str(i))
            onames.append(imgname[:-4])
        test_objs = []
        for i in range(len(onames)):
            oa = utils.generate_obj_attr(
                onames[i],
                imgs_path,
                scenario,
                scenario.width,
                scenario.height,
                total_time,
                distance_motion,
                n_step,
                obj_width,
                obj_height,
                reproduction_prohibition_time
            )

            nobj = utils.ImgObject(**oa)
            test_objs.append(nobj)

        #objs = [obj1, obj2, obj3]
        #objs = [obj1, obj2]
        objs = test_objs

        #for obj in objs:
        #    obj.set_scenario_knowledge(scenario.scenario_tuples)

        #res = utils.UpdateObjectScenarioKnowledge().operation(objs_list=[obj1],  scenario=scenario)
        #print(f"objs: {objs}")
        #utils.DreamingMachine.img_path_env = imgs_path

        #Start graphical environment
        screen, rimg = utils.start_pygame_env(scenario_width, scenario_height, obj_width, obj_height, rimg_path)
        print(f"screen: {screen}")
        #Start simulation
        res = utils.DreamingMachine.start_dreaming(
            objs_list=objs,
            scenario=scenario,
            n_step=n_step,
            i_step=0,
            img_path_env=imgs_path,
            simulation_time=total_time,
            distance_motion=distance_motion,
            body_width=obj_width,
            body_height=obj_height,
            time_vector=time_vector,
            dt=dt,
            screen=screen,
            rimg=rimg,
            graphical_env=True,
            verbose = False,
            reproduction_prohibition_time=reproduction_prohibition_time,
            max_intrcptd_objs=max_intrcptd_objs
        )
        #print(res)
        #print(f"scenario obj1: {res['objs_list'][0].c_scenario_knowledge}")

if __name__ == "__main__":
    SimulationTest.test_objectknowledge_setting()







