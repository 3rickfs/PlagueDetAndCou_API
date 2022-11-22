"""
 Test Brownian Motion module

"""
import matplotlib.pyplot as plt
import numpy as np

#print(f"sys.path: {sys.path}")

from IA_ModelsToolBox import BrownianMotion
#from Radon222DetAndCou_CVA_API.Radon222DetAndCou_CVA_API.api.IA_ModelsToolBox import BrownianMotion
#from . import BrownianMotion

class BrownianTest:

    @staticmethod
    def test_ploting_brownian_values():
        print("testing brownian motion lib")
        bv = []
        n_step = 100
        total_time = 10 #secs or days or weeks...
        time_vector = np.linspace(0,total_time,num=n_step)
        s0 = 2
        mu = 0.2
        sigma = 0.68
        b_objt = BrownianMotion.Brownian(n_step, time_vector, s0, mu, sigma)

        for i in range(len(time_vector)):
            bv.append(b_objt.geometric_brownian_motion(i))

        plt.plot(bv)
        plt.show()

if __name__ == "__main__":
    BrownianTest.test_ploting_brownian_values()
