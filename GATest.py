import numpy as np

class GATest:
    # Define limits in radians for [black, red, blue]
    MIN_JOINT_LIMITS = np.array([-1.57, -1.57, -2.0]) # -90°, -90°, -114.6°
    MAX_JOINT_LIMITS = np.array([1.57, 1.57, 1.0])   #  90°,  90°,   57.3°
    NUM_FRAMES = 300
    NUM_LEGS = 8
    SECTIONS = 3

    def ini_pol(self, n: int):
        rng = np.random.default_rng() 
        init_population = rng.uniform(
                            self.MIN_JOINT_LIMITS, 
                            self.MAX_JOINT_LIMITS, 
                            size=(n, self.NUM_FRAMES, self.NUM_LEGS, self.SECTIONS)
                        )

        return init_population

