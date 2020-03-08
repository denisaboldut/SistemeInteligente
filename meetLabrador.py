# -*- coding: utf-8 -*-
"""
Created on Thu Mar  5 20:18:59 2020

@author: boldutdenisa
"""

"""
most Labrador Retriver registrations are in the UK (18 554)- most likely place to meet one
    1/44 (contries in Europe) = 0.022
the weather is nice- most likely to meet one outside
    3/7 (3 days per week are sunny) = 0.428
"""

from pomegranate import *

location = DiscreteDistribution({True: 0.022, False: 0.978})
weather = DiscreteDistribution({True: 0.428, False: 0.572})
labrador = ConditionalProbabilityTable(
    [[True, True, True, 0.9],
     [True, True, False, 0.1],

     [True, False, True, 0.3],
     [True, False, False, 0.7],

     [False, True, True, 0.4],
     [False, True, False, 0.6],

     [False, False, True, 0.2],
     [False, False, False, 0.8]
     ],
    [location, weather])

s1 = State(location, name="Am I in the UK?")
s2 = State(weather, name="The weather is nice")
s3 = State(labrador, name="Meet a labrador")

model = BayesianNetwork()
model.add_states(s1, s2, s3)
model.add_edge(s1, s3)
model.add_edge(s2, s3)
model.bake()

print(model.probability([True,True,True]))
print(model.probability([True,True,False]))
print(model.probability([True,False,False]))