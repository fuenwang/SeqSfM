# -*- coding: utf-8 -*-

import os
import json
import errno
import pickle
import gzip
import numpy as np
import networkx as nx
import cv2

def load_tracks_graph(fileobj):
    g = nx.Graph()
    data = {}
    track_data = {}
    for line in fileobj:
        image, track, observation, x, y, R, G, B = line.split('\t')
        g.add_node(image, bipartite=0)
        g.add_node(track, bipartite=1)
        g.add_edge(image, track,
            feature=(float(x), float(y)),
            feature_id=int(observation),
            feature_color=(float(R), float(G), float(B)))
        try:
            if not int(observation) in data[image]:
                data[image].append(int(observation))
        except:
            data[image] = [int(observation)]
        try:
            track_data[image][observation] = track
        except:
            track_data[image] = {}
            track_data[image][observation] = track
    return g, data, track_data


def save_tracks_graph(fileobj, graph):
    for node, data in graph.nodes(data=True):
        if data['bipartite'] == 0:
            image = node
            for track, data in graph[image].items():
                x, y = data['feature']
                fid = data['feature_id']
                r, g, b = data['feature_color']
                fileobj.write('%s\t%s\t%d\t%g\t%g\t%g\t%g\t%g\n' % (str(image), str(track), fid, x, y, r, g, b))


def common_tracks(g, im1, im2):
    """
    Return the list of tracks observed in both images
    :param g: Graph structure (networkx) as returned by :method:`DataSet.tracks_graph`
    :param im1: Image name, with extension (i.e. 123.jpg)
    :param im2: Image name, with extension (i.e. 123.jpg)
    :return: tuple: track, feature from first image, feature from second image
    """
    # TODO: return value is unclear
    # TODO: maybe make as method of DataSet?
    t1, t2 = g[im1], g[im2]
    tracks, p1, p2 = [], [], []
    for track in t1:
        if track in t2:
            p1.append(t1[track]['feature'])
            p2.append(t2[track]['feature'])
            tracks.append(track)
    p1 = np.array(p1)
    p2 = np.array(p2)
    return tracks, p1, p2
