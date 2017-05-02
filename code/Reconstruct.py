import os
import sys
import cv2
import Config
import Bundle
import Feature
import numpy as np
import pyopengv as gv


if __name__ == '__main__':
    config = Config.Config(sys.argv[1])
    graph = Feature.Graph(config)
    graph.ConstructGraph()


