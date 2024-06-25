#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon Jun 17 11:23:55 2024

@author: emir
"""
#libraries
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt


#data import
datas = pd.read_csv("veriler.csv")
print(datas)


#data test
height = datas[["boy"]]
print(height)

height_weight = datas[["boy","kilo"]]
print(height_weight)