import os
import sys
import cv2
import Config
import Bundle
import Feature
import numpy as np
import pyopengv as gv

class Model:
    def __init__(self, config):
        self._config = config
        self._imglst = config.ImageList()
        self._graph = Feature.Graph(config)
        self._graph.ConstructGraph()
        self._feature_loc = {}
        for img in self._imglst:
            key = Config.ShortName(img)
            self._feature_loc[key] = config.LoadFeature(img)

    def GetGraph(self):
        return self._graph

    def Initialize(self):
        start_frame_lst = [Config.ShortName(x) for x in self._imglst]
        start_frame_lst = sorted(start_frame_lst, key=lambda x:len(self._graph.image_feature_graph[x]), reverse=True)


if __name__ == '__main__':
    config = Config.Config(sys.argv[1])
    model = Model(config)
    #g = model.GetGraph()
    #print g.image_image_graph
    model.Initialize()
