import os
import pandas as pd
import numpy as np





def compute_root_dir():
    root_dir = os.path.abspath(
        os.path.dirname(
        os.path.dirname(
            os.path.dirname(__file__))))
    return root_dir + os.path.sep


proj_root_dir = compute_root_dir()
# print(proj_rooy_dir)