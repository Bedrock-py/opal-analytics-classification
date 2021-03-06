#****************************************************************
# Copyright (c) 2015, Georgia Tech Research Institute
# All rights reserved.
#
# This unpublished material is the property of the Georgia Tech
# Research Institute and is protected under copyright law.
# The methods and techniques described herein are considered
# trade secrets and/or confidential. Reproduction or distribution,
# in whole or in part, is forbidden except by the express written
# permission of the Georgia Tech Research Institute.
#****************************************************************/

from bedrock.analytics.utils import * 

from sklearn.neighbors import KNeighborsClassifier
import numpy as np
import scipy as sp
import cPickle as pickle

class Knn(Algorithm):
    def __init__(self):
        super(Knn, self).__init__()
        self.parameters = []
        self.inputs = ['matrix.csv', 'truth_labels.csv']
        self.outputs = ['model']
        self.name ='KNN'
        self.type = 'Classification'
        self.description = 'Trains an KNN model using the input dataset with identified truth labels.'
        self.parameters_spec = [
             { "name" : "Algorithm", "attrname" : "algorithm", "value" : "auto", "type" : "select", "options": ["auto","ball_tree","kd_tree","brute"] },
             { "name" : "Leaf size", "attrname" : "leaf_size", "value" : "30", "type" : "input"}
        ]

    def compute(self, filepath, **kwargs):
        inputData = np.genfromtxt(filepath['matrix.csv']['rootdir'] + 'matrix.csv', delimiter=',') 
        if len(inputData.shape) == 1:
            inputData.shape=[inputData.shape[0],1]
        truth = np.genfromtxt(filepath['truth_labels.csv']['rootdir'] + 'truth_labels.csv', delimiter=',')
        model = KNeighborsClassifier(algorithm=self.algorithm, leaf_size=int(self.leaf_size))
        name = kwargs['name']
        model.fit(inputData, truth)
        modelpath = kwargs['storepath'] + name + '.p'
        pickle.dump( model, open( modelpath, "wb" ) )
        modelText = '''

from analytics.utils import Algorithm 
from sklearn.neighbors import KNeighborsClassifier
import scipy as sp
import numpy as np
import cPickle as pickle

class %s(Algorithm):
    def __init__(self):
        super(%s, self).__init__()
        self.parameters = []
        self.inputs = ['matrix.csv']
        self.outputs = ['assignments.csv']
        self.name ='%s'
        self.type = 'Classification'
        self.description = 'Applies the KNN model trained on a specific dataset: %s'
        self.parameters_spec = []
        self.modelfile = '%s'

    def compute(self, filepath, **kwargs):
        inputData = np.genfromtxt(filepath['matrix.csv']['rootdir'] + 'matrix.csv', delimiter=',') 
        if len(inputData.shape) == 1:
            inputData.shape=[inputData.shape[0],1]
        model = pickle.load( open( self.modelfile, "rb" ) )
        assignments = model.predict(inputData)
        self.results = {'assignments.csv': assignments}

    def classify(self, input):
        model = pickle.load( open( self.modelfile, "rb" ) )
        return model.predict(input)
''' % (name, name, name, filepath['matrix.csv']['rootdir'], modelpath)
        self.results = {'analytic': {'text': modelText, 'classname': name} }

