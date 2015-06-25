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

from analytics.utils import Algorithm 

from sklearn.neighbors import KNeighborsClassifier
import pandas as pd
import scipy as sp
import numpy as np
import cPickle as pickle


class Kkn(Algorithm):
    def __init__(self):
        super(Kkn, self).__init__()
        self.parameters = ['truthLabelsColumn','modelName']
        self.inputs = ['matrix.csv']
        self.outputs = ['model']
        self.name ='KKN'
        self.type = 'Classification'
        self.description = 'Trains an KKN model using the input dataset with identified truth labels.'
        self.parameters_spec = [ { "name" : "Column index of truth labels", "attrname" : "truthLabelsColumn", "value" : 0, "type" : "input", "step": 1, "max": 1000, "min": 0 },
                                { "name" : "Name of model", "attrname" : "modelName", "value" : "default", "type" : "input" }]

    def compute(self, filepath, **kwargs):
        inputData = np.genfromtxt(filepath['matrix.csv']['rootdir'] + 'matrix.csv', delimiter=',') 
        matrix = sp.delete(inputData, int(self.truthLabelsColumn), 1)
        labels = inputData[:,int(self.truthLabelsColumn)]
        model = KNeighborsClassifier()
        model.fit(matrix, labels)
        modelpath = kwargs['storepath'] + self.modelName + '.p'
        pickle.dump( model, open( modelpath, "wb" ) )

        modelText = '''

from analytics.utils import Algorithm 
from sklearn.neighbors import KNeighborsClassifier
import pandas as pd
import scipy as sp
import numpy as np
import cPickle as pickle

def get_classname():
    return '%s'

class %s(Algorithm):
    def __init__(self):
        super(%s, self).__init__()
        self.parameters = []
        self.inputs = ['matrix.csv']
        self.outputs = ['assignments.csv']
        self.name ='%s'
        self.type = 'Classification'

        self.description = 'Applies the KKN model trained on a specific dataset: %s'
        self.parameters_spec = []
        self.modelfile = '%s'

    def compute(self, filepath, **kwargs):
        inputData = np.genfromtxt(filepath['matrix.csv']['rootdir'] + 'matrix.csv', delimiter=',') 
        model = pickle.load( open( self.modelfile, "rb" ) )
        assignments = model.predict(inputData)
        self.results = {'assignments.csv': assignments}

''' % (self.modelName, self.modelName, self.modelName, self.modelName, filepath['matrix.csv']['rootdir'], modelpath)

        self.results = {'analytic': {'text': modelText, 'classname': self.modelName} }

