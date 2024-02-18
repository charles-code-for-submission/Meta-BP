import numpy as np
import scipy.sparse as sp
# import tensorflow as tf
# import yaml
# from pymongo import MongoClient


def is_binary_bag_of_words(features):
    features_coo = features.tocoo()
    return all(single_entry == 1.0 for _, _, single_entry in zip(features_coo.row, features_coo.col, features_coo.data))
