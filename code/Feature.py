import os
import sys
import cv2
import Config
import vlfeat
import numpy as np
import pyflann
import pyopengv
from multiprocessing import Pool


def normalized(loc, height, width):
    bigger = max(height, width)
    loc[:, 0] -= width / 2
    loc[:, 1] -= height / 2
    loc /= bigger
    return loc


def denormalized(loc, height, width):
    bigger = max(height, width)
    loc *= bigger
    loc[:, 0] += width / 2
    loc[:, 1] += height / 2
    return loc


def _extractWrap(buf):
    obj = buf[0]
    img_name = buf[1]
    [loc, des] = obj.Extract(img_name)
    obj.SaveFeature(img_name, loc, des)


def _matchWrap(buf):
    obj = buf[0]
    img = buf[1]
    lst = buf[2]
    print 'Match for %s' % img
    data = {}
    for to_match in lst:
        key = to_match.split('/')[-1]
        index = obj.Match_one_frame(img, to_match)
        data[key] = index

    obj.SaveMatch(img, data)


class Extractor:

    def __init__(self, config):
        self._config = config
        self._feature_path = config.FeaturePath()
        if not os.path.isdir(self._feature_path):
            os.system('mkdir -p %s' % self._feature_path)

    def Extract(self, img_name):
        img = cv2.imread(img_name, cv2.IMREAD_GRAYSCALE)
        img = img.astype(np.float32) / 255
        height = img.shape[0]
        width = img.shape[1]
        # print img
        peak = self._config.Get('sift_peak_threshold')
        edge = self._config.Get('sift_edge_threshold')
        loc, des = vlfeat.vl_sift(img, peak_thresh=peak, edge_thresh=edge)
        #loc, des = vlfeat.vl_sift(img)
        loc = np.round(loc[0:2, :].T)
        loc[:, 0] -= width / 2
        loc[:, 1] -= height / 2
        #bigger = max(width, height)
        #loc /= bigger
        des = des.T
        #self.SaveFeature(img_name, loc, des)
        print 'Extract SIFT %s total %d points' % (img_name, loc.shape[0])
        return [loc, des.astype(np.float32)]

    def SaveFeature(self, img_name, loc, des):
        short = img_name.split('/')[-1].split('.')[0]
        feature_path = '%s/%s.npy' % (self._feature_path, short)
        # print loc.shape
        # print des.shape
        np.save(feature_path, {'location': loc, 'descriptor': des})

    def Extract_all(self):
        lst = self._config.ImageList()
        arg = Config.ParameterTuple(self, lst)
        pool = Pool(processes=self._config.Get('threads'))
        # print arg
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
        result, dist = flann.nn(
            des2, des1, 2, algorithm="kmeans", branching=32, iterations=10, checks=200)

        index1 = np.arange(loc1.shape[0])
        compare = (dist[:, 0].astype(np.float32) / dist[:, 1]
                   ) < self._config.Get('flann_threshold')

        index1 = index1[compare]
        index2 = result[:, 0][compare]

        loc1 = loc1[index1, :]
        loc2 = loc2[index2, :]

        [F1, M1] = cv2.findFundamentalMat(loc1, loc2, cv2.FM_RANSAC)
        [F2, M2] = cv2.findFundamentalMat(loc2, loc1, cv2.FM_RANSAC)
        M = M1 * M2
        M = np.reshape(M, [-1])

        index1 = index1[M == 1]
        index2 = index2[M == 1]

        return np.vstack([index1, index2]).T

    def Visual_Match(self, frame1, frame2):
        feature1 = self._config.LoadFeature(frame1)
        feature2 = self._config.LoadFeature(frame2)

        [loc1, des1] = [feature1['location'], feature1['descriptor']]
        [loc2, des2] = [feature2['location'], feature2['descriptor']]

        flann = pyflann.FLANN()
        result, dist = flann.nn(
            des2, des1, 2, algorithm="kmeans", branching=32, iterations=10, checks=200)

        index1 = np.arange(loc1.shape[0])
        compare = (dist[:, 0].astype(np.float32) / dist[:, 1]
                   ) < self._config.Get('flann_threshold')

        index1 = index1[compare]
        index2 = result[:, 0][compare]

        loc1 = loc1[index1, :]
        loc2 = loc2[index2, :]

        [F1, M1] = cv2.findFundamentalMat(loc1, loc2, cv2.FM_RANSAC)
        [F2, M2] = cv2.findFundamentalMat(loc2, loc1, cv2.FM_RANSAC)
        M = M1 * M2
        M = np.reshape(M, [-1])
        loc1 = loc1[M == 1, :]
        loc2 = loc2[M == 1, :]

        img1 = cv2.imread(frame1, cv2.IMREAD_COLOR)
        img2 = cv2.imread(frame2, cv2.IMREAD_COLOR)

        height = img1.shape[0]
        width = img1.shape[1]

        big = np.zeros([height, 2 * width, 3], dtype=np.uint8)
        big[:, 0:width] = img1
        big[:, width:] = img2

        # bigger = max(width, height])
        #loc1 *= bigger
        #loc2 *= bigger
        loc1[:, 0] += width / 2
        loc1[:, 1] += height / 2
        loc2[:, 0] += width / 2
        loc2[:, 1] += height / 2

        loc1 = loc1.astype(int)
        loc2[:, 0] += width
        loc2 = loc2.astype(int)
        for i in range(loc1.shape[0]):
            cv2.line(big, (loc1[i, 0], loc1[i, 1]),
                     (loc2[i, 0], loc2[i, 1]), (0, 255, 0))
        cv2.namedWindow('Visualize')
        cv2.imshow('Visualize', big)
        cv2.waitKey(0)

    def Pose_test(self, frame1, frame2):
        feature1 = self._config.LoadFeature(frame1)
        feature2 = self._config.LoadFeature(frame2)

        [loc1, des1] = [feature1['location'], feature1['descriptor']]
        [loc2, des2] = [feature2['location'], feature2['descriptor']]

        flann = pyflann.FLANN()
        result, dist = flann.nn(
            des2, des1, 2, algorithm="kmeans", branching=32, iterations=10, checks=200)

        index1 = np.arange(loc1.shape[0])
        compare = (dist[:, 0].astype(np.float32) / dist[:, 1]
                   ) < self._config.Get('flann_threshold')

        index1 = index1[compare]
        index2 = result[:, 0][compare]

        loc1 = loc1[index1, :]
        loc2 = loc2[index2, :]

        [F1, M1] = cv2.findFundamentalMat(loc1, loc2, cv2.FM_RANSAC)
        [F2, M2] = cv2.findFundamentalMat(loc2, loc1, cv2.FM_RANSAC)
        M = M1 * M2
        M = np.reshape(M, [-1])
        loc1 = loc1[M == 1, :]
        loc2 = loc2[M == 1, :]

        loc1 = normalized(loc1, 720, 1280)
        loc2 = normalized(loc2, 720, 1280)

        a = np.ones([loc1.shape[0], 3], np.float)
        a[:, :2] = loc1
        b = np.ones([loc1.shape[0], 3], np.float)
        b[:, :2] = loc2
        # print a
        # eight point is better
        M = pyopengv.relative_pose_eightpt(a, b)
        # print M
        i = 20
        print a[i, :]
        print b[i, :]
        print np.dot(np.dot(a[i, :], M), b[i, :].T)
        '''
        M = pyopengv.relative_pose_fivept_nister(a, b)
        print M
        '''
        # print a[0,:]
        # print b[0,:]
        # print np.dot(M, a[0, :])

    def Match_all(self):
        nframes = self._config.Get('nframes')
        match_lst = []
        total_frames = len(self._imglst)
        for i, img in enumerate(self._imglst):
            start = i + 1
            if start >= total_frames:
                match_lst.append([])
                break
            if i + nframes < total_frames:
                end = i + nframes
            else:
                end = total_frames - 1

            to_add = self._imglst[start:end + 1]
            match_lst.append(to_add)

        arg = []
        for i, one in enumerate(match_lst):
            arg.append([self, self._imglst[i], one])

        pool = Pool(processes=self._config.Get('threads'))
        pool.map(_matchWrap, arg)

    def SaveMatch(self, img, data):
        if not os.path.isdir(self._config.MatchPath()):
            os.system('mkdir %s' % self._config.MatchPath())
        name = img.split('/')[-1].split('.')[0]
        name = '%s/%s.npy' % (self._config.MatchPath(), name)
        np.save(name, data)


class Graph:

    def __init__(self, config):
        self._config = config
        self._imglst = config.ImageList()

        self.track_graph = {}
        self.image_feature_graph = {}
        self.image_image_graph = {}

    def ConstructGraph(self):
        track_graph = self.track_graph  # A[track][img]
        # image_track_graph = self.image_track_graph # A[img][track]
        image_feature_graph = self.image_feature_graph  # A[img][feature_id]
        image_image_graph = self.image_image_graph
        match_data = {}
        for img in self._imglst:
            name = Config.ShortName(img)
            match = self._config.LoadMatch(img)
            match_data[name] = match

        next_track_id = 1
        for img_full in self._imglst:
            img = Config.ShortName(img_full)
            match = match_data[img]
            match_frames = match.keys()
            '''
            if img not in image_track_graph:
                image_track_graph[img] = {}
            '''
            if img not in image_feature_graph:
                image_feature_graph[img] = {}
            if img not in image_image_graph:
                image_image_graph[img] = {}

            for frame in match_frames:
                frame_match = match[frame]
                if frame not in image_feature_graph:
                    image_feature_graph[frame] = {}
                if frame not in image_image_graph:
                    image_image_graph[frame] = {}

                for pair in frame_match:
                    # if wee find a new track
                    if pair[0] not in image_feature_graph[img] and pair[1] not in image_feature_graph[frame]:
                        image_feature_graph[img][pair[0]] = next_track_id
                        image_feature_graph[frame][pair[1]] = next_track_id
                        if next_track_id not in track_graph:
                            track_graph[next_track_id] = {}
                        track_graph[next_track_id][img] = pair[0]
                        track_graph[next_track_id][frame] = pair[1]

                        if frame not in image_image_graph[img]:
                            image_image_graph[img][frame] = []
                        if img not in image_image_graph[frame]:
                            image_image_graph[frame][img] = []
                        image_image_graph[img][frame].append(next_track_id)
                        image_image_graph[frame][img].append(next_track_id)
                        next_track_id += 1

                    elif pair[0] not in image_feature_graph[img]:
                        track_id = image_feature_graph[frame][pair[1]]
                        image_feature_graph[img][pair[0]] = track_id
                        track_graph[track_id][img] = pair[0]
                        if frame not in image_image_graph[img]:
                            image_image_graph[img][frame] = []
                        if img not in image_image_graph[frame]:
                            image_image_graph[frame][img] = []
                        image_image_graph[img][frame].append(track_id)
                        image_image_graph[frame][img].append(track_id)

                    elif pair[1] not in image_feature_graph[frame]:
                        track_id = image_feature_graph[img][pair[0]]
                        image_feature_graph[frame][pair[1]] = track_id
                        track_graph[track_id][frame] = pair[1]
                        if frame not in image_image_graph[img]:
                            image_image_graph[img][frame] = []
                        if img not in image_image_graph[frame]:
                            image_image_graph[frame][img] = []
                        image_image_graph[img][frame].append(track_id)
                        image_image_graph[frame][img].append(track_id)


if __name__ == '__main__':
    config = Config.Config(sys.argv[1])
    lst = config.ImageList()
    #extract = Extractor(config)
    # extract.Extract_all()
    #'''
    match = Matcher(config)
    #match.Visual_Match(lst[-1], lst[-3])
    match.Pose_test(lst[-1], lst[-3])
    # match.Match_all()
    #'''
    '''
    graph = Graph(config)
    graph.ConstructGraph()
    print graph.track_graph
    '''
