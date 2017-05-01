import os
import yaml
import numpy as np

class Config:

    def __init__(self, path):
        self._root = path
        self._img_lst = [x for x in sorted(os.listdir(path + '/images'))]
        self._img_lst = ['%s/images/%s'%(path,x) for x in self._img_lst]
        self._feature_path = '%s/feature'%path
        self._match_path = '%s/match'%path

        f = open('%s/config.yaml'%path)
        self._set = yaml.load(f)
        f.close()

    def ImageList(self):
        return self._img_lst

    def FeaturePath(self):
        return self._feature_path

    def MatchPath(self):
        return self._match_path

    def Get(self, key):
        return self._set[key]
    
    def LoadFeature(self, img_name):
        shortname = img_name.split('/')[-1].split('.')[0]
        name = '%s/%s.npy'%(self._feature_path, shortname)
        return np.load(name).item()

    def LoadMatch(self, img_name):
        shortname = img_name.split('/')[-1].split('.')[0]
        name = '%s/%s.npy'%(self._match_path, shortname)
        return np.load(name).item()

def ParameterTuple(first, second): # [first, second[0]], ....
    data = []
    for one in second:
        data.append([first, one])

    return data

def ShortName(img):
    return img.split('/')[-1]
