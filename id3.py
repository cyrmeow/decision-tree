from collections import namedtuple
import sys
import math
import numpy as np
import scipy.stats as st

from Data import *


DtNode = namedtuple("DtNode", "fVal, nPosNeg, gain, left, right")

POS_CLASS = 'e'

def Entropy(data):
    # calculate the entropy of dataset data
    # param data: dataset
    # return: entropy value
    if len(data) == 0:
        return 0
    nPos = sum(d[0] == POS_CLASS for d in data)
    pPos = float(nPos) / len(data)
    if pPos == 1 or pPos == 0:
        return 0
    else:
        pNeg = 1 - pPos
        return -pPos * np.log(pPos) / np.log(2) - pNeg * np.log(pNeg) / np.log(2) 

def InformationGain(data, f):
    #TODO: compute information gain of this dataset after splitting on feature F
    trueData = [d for d in data if d[f.feature] == f.value]
    falseData = [d for d in data if d[f.feature] != f.value]
    conditional_entropy = Entropy(trueData) * (float(len(trueData)) / len(data)) + Entropy(falseData) * (float(len(falseData)) / len(data))
    return Entropy(data) - conditional_entropy

def Classify(tree, instance):
    if tree.left == None and tree.right == None:
        return tree.nPosNeg[0] > tree.nPosNeg[1]
    elif instance[tree.fVal.feature] == tree.fVal.value:
        return Classify(tree.left, instance)
    else:
        return Classify(tree.right, instance)

def Accuracy(tree, data):
    nCorrect = 0
    for d in data:
        if Classify(tree, d) == (d[0] == POS_CLASS):
            nCorrect += 1
    return float(nCorrect) / len(data)

def PrintTree(node, prefix=''):
    print("%s>%s\t%s\t%s" % (prefix, node.fVal, node.nPosNeg, node.gain))
    if node.left != None:
        PrintTree(node.left, prefix + '-')
    if node.right != None:
        PrintTree(node.right, prefix + '-')        
        
def ID3(data, features, MIN_GAIN=0.1):
    #TODO: implement decision tree learning

    #step 1. find the feature that gives the largest information gain
    maxig = 0
    for f in features:
        if maxig <= InformationGain(data, f):
            maxig = InformationGain(data, f)
            split_feature = f

    #step 2. if maxig is less than threshold, stop growing and return it as a leaf node
    if(maxig <= MIN_GAIN):
        nPos = sum(d[0] == POS_CLASS for d in data)
        nNeg = len(data) - nPos
        return DtNode(split_feature, (nPos, nNeg), maxig, None, None)

    #step 3. otherwise, grow the tree recursively
    else:
        trueData = [d for d in data if d[split_feature.feature] == split_feature.value]
        falseData = [d for d in data if d[split_feature.feature] != split_feature.value]
        nPos = sum(d[0] == POS_CLASS for d in data)
        nNeg = len(data) - nPos
        features.remove(split_feature)
        return DtNode(split_feature, (nPos, nNeg), maxig, ID3(trueData, features, MIN_GAIN), ID3(falseData, features, MIN_GAIN))

if __name__ == "__main__":


#
    train = MushroomData(sys.argv[1])

    dev = MushroomData(sys.argv[2])

    dTree = ID3(train.data, train.features, MIN_GAIN=float(sys.argv[3]))
    
    PrintTree(dTree)

    print Accuracy(dTree, dev.data)
