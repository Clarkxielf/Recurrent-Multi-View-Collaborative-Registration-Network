import numpy as np

def RandomJitter(pts, scale=0.0, clip=0.5):
    """ generate perturbations """

    noise = np.clip(np.random.normal(0.0, scale=scale, size=pts.shape), a_min=-clip, a_max=clip)
    pts += noise  # Add noise to xyz

    return pts