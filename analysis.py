#!/usr/bin/env python

import sys
sys.path.insert(0,'/Users/cmdb/projects/ConfocalAnalysis')
from czifile import CziFile
import numpy as np
from math import sqrt
from skimage.feature import blob_log
import matplotlib.pyplot as plt
from matplotlib import gridspec
from sklearn.cluster import DBSCAN
from sklearn import metrics
from mpl_toolkits.mplot3d import Axes3D
import operator

"""Useful Functions"""

def GetOneChannel( czi_file, channel ):
# this function takes the whole .czi file and return a single channel .czi
    stacknumber=czi_file.shape[3]
    OneChannel=[czi_file[0,channel,0,index].T[0].T for index in range(stacknumber)]
    return OneChannel

def FindBlobs( OneChannel ):
# this function takes output from GetOneChannel (a single channel .czi file input)
# return all the blobs in a dict & a structured list & a flattened list
# Dict    # "Y,X,Z"->[radius, intensity]
# struct. List    # [[[Y,X,Z1],[Y,X,Z1]...],[[Y,X,Z2],[Y,X,Z2],...],...,[[Y,X,Zn],...]],np.array
# flat. List    # [[Y1,X1,Z1],[Y2,X2,Z2],[Y3,X3,Z3],...,[Yn,X3,Z3]]
  
    blobs_dict={}
    blobsAll=[]
    blobsList=[]
    stacknumber=len(OneChannel)
    
    for z in range(stacknumber):
        planeblobs=[]
        """Blob Recognition Parameters here"""
        log = blob_log(OneChannel[z], max_sigma=30, num_sigma=10, threshold=.2)
        # Radius of the blob
        log[:, 2] = log[:, 2] * sqrt(2)
        # intensity
        blob_intensity=np.zeros(len(log[:,0]))
        
        for i in range(len(log[:,0])):
            blob_intensity[i]=OneChannel[z][int(log[i,0]),int(log[i,1])]
        
        for i in range(len(log[:,2])):
            position=np.append(log[i,0:2],z)
            position=map(int,position)
            blobs_dict["{},{},{}".format(position[0],position[1],position[2])]=[log[i,2],blob_intensity[i]]
            planeblobs.append(position)
        
        blobsAll.append(planeblobs)
    
    blobsAll=np.array(blobsAll)

    for z in range(len(blobsAll)):
        blobsList=blobsList+blobsAll[z]
    blobsList=np.array(blobsList)
    
    return blobs_dict,blobsAll,blobsList



def GetClustersOneChannel( blobsList ):
# this function takes the dict and flattened list from FindBlobs
# return the clusters
###  [[[cluster1_dot1],[cluster1_dot2],...],
#     [[cluster2_dot1],[cluster2_dot2]...],
#     [...],...,
#     [[clusterN_dot1],[clusterN_dot2]] ]

    """3D clustering of Blobs"""
    db = DBSCAN(eps=2, min_samples=2).fit(blobsList)
    labels = db.labels_
    # Number of clusters in labels, ignoring noise if present.
    n_clusters_ = len(set(labels)) - (1 if -1 in labels else 0)

    print('Estimated number of clusters: %d' % n_clusters_)
    print("Silhouette Coefficient: %0.3f"
          % metrics.silhouette_score(blobsList, labels))
    
    """Group up the dots within a cluster"""
    clusters=[]
    for i in range(n_clusters_):
        cluster=[]
        ## Extract X,Y,Z location of the points based on clusters
        positions=np.argwhere(labels==i)
        for j in range(len(positions)):
            cluster.append(blobsList[positions[j]])
        clusters.append(cluster)

    return clusters

def SignalPositionsOneChannel( clusters,blobs_dict ):
# This function takes the clusters List from GetClustersOneChannel
# return the calculated signal position in this one channel all stacks
# [[Y,X,Z,avgIntensity],[Y,X,Z,avgIntensity],...,[Y,X,Z,avgIntensity]]
    """Find the actual signal location"""
    signals=[]
    for i in range(len(clusters)):
        signal=[]
        dots=[]
        ## extract the intensity of the dot in the cluster (from the dictionary)
        for j in range(len(clusters[i])):
            #print clusters[i][j][0]
            dot=clusters[i][j][0]
            intensity=blobs_dict["{},{},{}".format(clusters[i][j][0][0],clusters[i][j][0][1],clusters[i][j][0][2])][1]
            dot=np.append(dot,intensity)
            dots.append(dot)
        # Calculating the actual dot location
        intensity_sum=sum(dots[index][3] for index in range(len(dots)))
        avgIntensity=intensity_sum/len(dots)
        y=sum(dots[index][0]*dots[index][3]/intensity_sum for index in range(len(dots)))
        x=sum(dots[index][1]*dots[index][3]/intensity_sum for index in range(len(dots)))
        z=sum(dots[index][2]*dots[index][3]/intensity_sum for index in range(len(dots)))
        signal=[y,x,z,avgIntensity]
        signals.append(signal)
    signals=np.array(signals)
    return signals

def ToActualScale(list):
# This function takes a [n_sample,[Y,X,Z,Intensity]] LIST
# convert to the actual scale of the coordinates
    """Adjust Parameters"""
    # 63X objective, unit: um
    x_scale=0.099
    y_scale=0.099
    z_scale=0.2
    actual=np.zeros((len(list),4))
    for i in range(len(list)):
        actual[i][0]=list[i][0]*y_scale
        actual[i][1]=list[i][1]*x_scale
        actual[i][2]=list[i][2]*z_scale
        actual[i][3]=list[i][3]
    return actual

def GetDistance(element1,element2):
# This function takes two [Y,X,Z]
# return euclidean distance
    distance=sqrt((element1[0]-element2[0])**2+(element1[1]-element2[1])**2+(element1[2]-element2[2])**2)
    return distance

def FindPairs( signals1,signals2 ):
# This function takes 2 signals-Lists generated in SignalPositionsOneChannel
# return pairs  (LIST)
# [[sig1_1,sig2_N1],[sig1_2,sig2_N2],...,[sig1_N,sig2_Nn]]
    pairs=[]
    signals1=ToActualScale(signals1)
    signals2=ToActualScale(signals2)
    #print signals1
    #print signals2
    # empirically, distance between 2 dots won't exceed 1.2 microns
    """Adjust Parameters"""
    limit=1.2
    
    # Brute-Force solve
    for i in range(len(signals1)):
        distances=[]
        for j in range(len(signals2)):
            if signals2[j][0]==0:
                distances.append(100)  # rule out the past dots
                continue
            distance=GetDistance(signals1[i],signals2[j])
            distances.append(distance)
        min_index, min_value = min(enumerate(distances), key=operator.itemgetter(1))
        if min_value>limit:
            continue
        pairs.append([i,min_index,min_value])
        signals2[min_index]=[0,0,0,0]
#         print signals2[min_index]
#         print signals2
    return pairs







# Main function
""" Image Input """
with CziFile('/Users/cmdb/data/Image_1.czi') as czi:
    image_arrays=czi.asarray()
# 63X objective, unit: um
x_scale=0.099
y_scale=0.099
z_scale=0.2
"""Image Processing"""
print 'input done'
# Get one channel for ss
OneChannel_ss=GetOneChannel(image_arrays,2)
blobs_dict_ss,blobsAll_ss,blobsList_ss=FindBlobs(OneChannel_ss)
# Get one channel for klu
OneChannel_klu=GetOneChannel(image_arrays,3)
blobs_dict_klu,blobsAll_klu,blobsList_klu=FindBlobs(OneChannel_klu)
print 'blobs done'
"""Analysis the blobs"""
clusters_ss=GetClustersOneChannel(blobsList_ss)
clusters_klu=GetClustersOneChannel(blobsList_klu)

#print clusters_ss
signals_ss=SignalPositionsOneChannel(clusters_ss,blobs_dict_ss)
signals_klu=SignalPositionsOneChannel(clusters_klu,blobs_dict_klu)
print 'clusters done'
#######plot the two signals
# y_s,x_s,z_s,i_s = signals_ss.T
# y_k,x_k,z_k,i_k = signals_klu.T
#
# fig = plt.figure()
# ax = fig.add_subplot(111, projection='3d')
#
# ax.scatter(x_s*x_scale, y_s*y_scale, z_s*z_scale,color='green')
# ax.scatter(x_k*x_scale, y_k*y_scale, z_k*z_scale,color='red')
#
# ax.set_xlabel('X Label')
# ax.set_ylabel('Y Label')
# ax.set_zlabel('Z Label')
# ax.set_zlim([-1,6])
#
# plt.show()

# print len(blobsList)
# print blobsList


pairs=FindPairs(signals_ss,signals_klu)
print 'Pairs Done'
# extract the long ones
long=[]
for pair in pairs:
    #print pair
    if pair[2]>0.8:
        long.append(pair[0:3])

pair_coord=[]
for pair in long:
    pair_coord.append([signals_ss[pair[0]][0:3],signals_klu[pair[1]][0:3]])
    
print 'coordinate done'  
## 3D plot

#print pair_coord
y_s,x_s,z_s = np.array([pair_coord[index][0] for index in range(len(pair_coord))]).T
y_k,x_k,z_k = np.array([pair_coord[index][1] for index in range(len(pair_coord))]).T

fig = plt.figure()
ax = fig.add_subplot(111, projection='3d')

ax.scatter(x_s, y_s, z_s,color='green')
ax.scatter(x_k, y_k, z_k,color='red')

ax.set_xlabel('X Label')
ax.set_ylabel('Y Label')
ax.set_zlabel('Z Label')
ax.set_zlim([-5,30])

plt.show()