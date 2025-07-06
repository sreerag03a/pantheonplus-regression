import os
import sys
import numpy as np
import pandas as pd
from src.components.handling.exceptions import CustomException
import dill


def save_obj(filepath,obj):

    try:
        dir_path = os.path.dirname(filepath)
        os.makedirs(dir_path,exist_ok=True)

        with open(filepath,"wb") as fileobj:
            dill.dump(obj, fileobj)

    except Exception as e:
        raise CustomException(e,sys)