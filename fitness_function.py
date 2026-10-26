
import numpy as np

def fitness_function(
    gait,
    forward_distance,
    target_distance,
    stability_margins,
    feet_on_ground
):
    #1. Reward: forward progress
     

    # 2. Smoothness penalty
    # Smoothness penalty measures how smooth the spider moves per frame 
    changes = np.diff(gait, axis=0)
    smoothness_penalty = np.std(changes)
    #!!! Important notice, If the change is really big, the smoothness penalty would still be 0.

    # 3. Static stability penalty
    # Positive margin = inside the support polygon
    # Negative margin = outside the support polygon
    required_margin = 0.05 

    stability_penalty =  np.maximum(0, required_margin - stability_margins) ** 2

    #How do we get the stability margin???
    

    # 4. Feet-on-ground penalty
    # Penalise steps with fewer than 3 feet on the ground
    minimum_feet = 3

    feet_penalty = np.mean(
        np.maximum(0, minimum_feet - feet_on_ground)
    )

    # Things to note, for this fitness penalty to work, we would need to have more than just the gait/individual (300 X 24) we would need the amount of feet on the ground,
    #  and the stability margin

    # 5. Combine reward and penalties
    fitness = (
        progress_reward
        - 0.2 * smoothness_penalty
        - 0.3 * stability_penalty
        - 0.2 * feet_penalty
    )

    return fitness
