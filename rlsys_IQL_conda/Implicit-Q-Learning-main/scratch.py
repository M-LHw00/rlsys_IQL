import torch
import pickle
import numpy as np
# import pandas as pd

def nothing():
    path = "adr_data/monotonic/0.pkl"

    with open(path,'rb') as f:
        buffer = pickle.load(f)

    print(type(buffer[0]))


if __name__== "__main__":
    nothing()
