#Copyright (c) 2015, Georgia Tech Research Institute
# All rights reserved.
#
# This unpublished material is the property of the Georgia Tech
# Research Institute and is protected under copyright law.
# The methods and techniques described herein are considered
# trade secrets and/or confidential. Reproduction or distribution,
# in whole or in part, is forbidden except by the express written
# permission of the Georgia Tech Research Institute.
#****************************************************************/

from analytics.utils import *

from sklearn import svm
import pandas as pd
import scipy as sp
import numpy as np
import cPickle as pickle


class Svm(Algorithm):
    def __init__(self):
        super(Svm, self).__init__()
        self.parameters = []
        self.inputs = ['matrix.csv', 'truth_labels.csv']
        self.outputs = ['model']
        self.name ='SVM'
        self.type = 'Classification'
        self.description = 'Trains an SVM model using the input dataset and truth labels.'
        self.parameters_spec = []

    def compute(self, filepath, **kwargs):
        inputData = np.genfromtxt(filepath['matrix.csv']['rootdir'] + 'matrix.csv', delimiter=',') 
        truth = np.genfromtxt(filepath['truth_labels.csv']['rootdir'] + 'truth_labels.csv', delimiter=',')
        name = kwargs['name']
        model = svm.SVC()
        model.fit(inputData, truth)
        modelpath = kwargs['storepath'] + name + '.p'
        pickle.dump( model, open( modelpath, "wb" ) )

        modelText = '''

from analytics.utils import Algorithm 
from sklearn import svm
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
        self.type = 'Model'

        self.description = 'Applies the SVM model trained on a specific dataset: %s'
        self.parameters_spec = []
        self.modelfile = '%s'

    def compute(self, filepath, **kwargs):
        inputData = np.genfromtxt(filepath['matrix.csv']['rootdir'] + 'matrix.csv', delimiter=',') 
        model = pickle.load( open( self.modelfile, "rb" ) )
        assignments = model.predict(inputData)
        self.results = {'assignments.csv': assignments}

    def classify(self, input):
        model = pickle.load( open( self.modelfile, "rb" ) )
        inputData = input.split(',')
        return model.predict(inputData)

''' % (name, name, name, name, filepath['matrix.csv']['rootdir'], modelpath)

        self.results = {'analytic': {'text': modelText, 'classname': name} }


