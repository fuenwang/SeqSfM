import json
import numpy as np
import dataset
import Bundle
import networkx as nx

f = open('tracks.csv')
[g, _, _] = dataset.load_tracks_graph(f)

f.close()

top, bottom = nx.algorithms.bipartite.sets(g)
#print g['1832']

f = open('reconstruction.json')
data = json.load(f)[0]
f.close()

f = data['cameras']['v2 unknown unknown -1 -1 perspective 0']['focal']
k1 = data['cameras']['v2 unknown unknown -1 -1 perspective 0']['k1']
k2 = data['cameras']['v2 unknown unknown -1 -1 perspective 0']['k2']

print f, k1, k2

bundle = Bundle.Bundle()

shots = data['shots']
points = data['points']
bundle.SetCamera(f, k1, k2)
for shot in shots:
    R = shots[shot]['rotation']
    T = shots[shot]['translation']
    bundle.AddCamera_tag(shot, R, T)

for track_id in points:
    [x, y, z] = points[track_id]['coordinates']
    bundle.Add3DPoint_tag(track_id, x, y, z)
    for shot in g[track_id]:
        if shot in shots:
            [x, y] = g[track_id][shot]['feature']
            bundle.Add2DPoint_tag(shot, track_id, x, y)

bundle.Merge()
bundle.Run()
np.save('res.npy', bundle.res)

for shot in shots:
    [R, T] = bundle.GetShot(shot)
    shots[shot]['rotation'] = R.tolist()
    shots[shot]['translation'] = T.tolist()


for track_id in points:
    loc = bundle.GetPoint(track_id)
    #loc[-1] *= -1
    points[track_id]['coordinates'] = loc.tolist()

[f, k1, k2] = bundle.GetCameraIntrinsic()
print f, k1, k2
data['cameras']['v2 unknown unknown -1 -1 perspective 0']['focal'] = f
data['cameras']['v2 unknown unknown -1 -1 perspective 0']['k1'] = k1
data['cameras']['v2 unknown unknown -1 -1 perspective 0']['k2'] = k2

f = open('new.json', 'w')
f.write(json.dumps([data], indent=4))
f.close()
