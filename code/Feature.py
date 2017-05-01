import os
import sys
import cv2
import Config
import vlfeat
import numpy as np
import pyflann
from multiprocessing import Pool

def _extractWrap(buf):
    obj = buf[0]
    img_name = buf[1]
    [loc, des] = obj.Extract(img_name)
    obj.SaveFeature(img_name, loc, des)

def _matchWrap(buf):
    obj = buf[0]
    img = buf[1]
    lst = buf[2]
    print 'Match for %s' %img   
    data = {}
    for to_match in lst:
        key = to_match.split('/')[-1]
        index =  obj.Match_one_frame(img, to_match)
        data[key] = index

    obj.SaveMatch(img, data)

class Extractor:
    def __init__(self, config):
        self._config = config
        self._feature_path = config.FeaturePath()
        if not os.path.isdir(self._feature_path):
            os.system('mkdir -p %s'%self._feature_path)

    def Extract(self, img_name):
        print 'Extract SIFT %s'%img_name
        img = cv2.imread(img_name, cv2.IMREAD_GRAYSCALE)
        img = img.astype(np.float32) / 255
        #print img
        peak = self._config.Get('sift_peak_threshold')
        edge = self._config.Get('sift_edge_threshold')
        loc, des = vlfeat.vl_sift(img, peak_thresh=peak, edge_thresh=edge)
        #loc, des = vlfeat.vl_sift(img)
        loc = np.round(loc[0:2, :].T)
        des = des.T
        #self.SaveFeature(img_name, loc, des)
        return [loc, des.astype(np.float32)]

    def SaveFeature(self, img_name, loc, des):
        short = img_name.split('/')[-1].split('.')[0]
        feature_path = '%s/%s.npy'%(self._feature_path, short)
        #print loc.shape
        #print des.shape
        np.save(feature_path, {'location':loc, 'descriptor':des})
    
    def Extract_all(self):
        lst = self._config.ImageList()
        arg = Config.ParameterTuple(self, lst)
        pool = Pool(processes = self._config.Get('threads'))
        #print arg
        pool.map(_extractWrap, arg)

class Matcher:
    def __init__(self, config):
        self._config = config

        self._imglst = self._config.ImageList()

    def Match_one_frame(self, frame1, frame2):
        feature1 = self._config.LoadFeature(frame1)
        feature2 = self._config.LoadFeature(frame2)

        [loc1, des1] = [feature1['location'], feature1['descriptor']]
        [loc2, des2] = [feature2['location'], feature2['descriptor']]

        flann = pyflann.FLANN()
        result, dist = flann.nn(des2, des1, 2, algorithm="kmeans", branching=32, iterations=10, checks=200)

        index1 = np.arange(loc1.shape[0])
        compare = (dist[:, 0].astype(np.float32) / dist[:, 1]) < self._config.Get('flann_threshold')

        index1 = index1[compare]
        index2 = result[:, 0][compare]

        return np.vstack([index1, index2]).T
    
    def Match_all(self):
        nframes = self._config.Get('nframes')
        match_lst = []
        total_frames = len(self._imglst)
        for i, img in enumerate(self._imglst):
            start = i + 1
            if start  >= total_frames:
                match_lst.append([])
                break
            if i + nframes < total_frames:
                end = i + nframes
            else:
                end = total_frames - 1

            to_add = self._imglst[start:end+1]
            match_lst.append(to_add)
        
        arg = []
        for i, one in enumerate(match_lst):
            arg.append([self, self._imglst[i], one])

        pool = Pool(processes = self._config.Get('threads'))
        pool.map(_matchWrap, arg)
        

    def SaveMatch(self, img, data):
        if not os.path.isdir(self._config.MatchPath()):
            os.system('mkdir %s'%self._config.MatchPath())
        name = img.split('/')[-1].split('.')[0]
        name = '%s/%s.npy'%(self._config.MatchPath(), name)
        np.save(name, data)

if __name__ == '__main__':
    config = Config.Config(sys.argv[1])
    lst = config.ImageList()
    #extract = Extractor(config)
    #extract.Extract_all()

    match = Matcher(config)

    match.Match_all()





