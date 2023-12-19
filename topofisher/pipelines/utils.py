#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Fri Sep 22 10:11:24 2023

@author: karthikviswanathan
"""

import pickle

def writeToFile(lis, fileName):
    g = open(fileName, "wb")
    pickle.dump(lis, g)
    g.close()

def readFromFile(fileName) :
    f = open(fileName, "rb")
    PD = pickle.load(f)
    f.close()
    return PD

