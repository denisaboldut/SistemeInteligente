# -*- coding: utf-8 -*-
"""
Created on Thu Mar  5 20:18:59 2020

@author: boldutdenisa
"""

"""
--------task1------------
most Labrador Retriver registrations are in the UK (18 554)- most likely place to meet one
    1/44 (contries in Europe) = 0.022
the weather is nice- most likely to meet one outside
    3/7 (3 days per week are sunny) = 0.428
    
--------task2-----------
labradors can be:
  - pet
  - guide dog 
  
1 of 1000 children are born blind
   1/1000 = 0.001
1 of 800 people suffering from blindness caused by accident 
   1/800 = 0.00125 
 
 meeting a blind man 
     increase the probability of meeting a guide dog   
 
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

pet = ConditionalProbabilityTable(
    [
        [True, True, 0.7],
        [True, False, 0.3],

        [False, True, 0.3],
        [False, False, 0.7],
    ],
    [labrador])

newbornBlindness = DiscreteDistribution({True: 0.001, False: 0.999})
accidentBlindness = DiscreteDistribution({True: 0.00125, False: 0.99875})
blindMan = ConditionalProbabilityTable(
    [
        [True, True, True, 0.999],
        [True, True, False, 0.001],

        [True, False, True, 0.98],
        [True, False, False, 0.02],

        [False, True, True, 0.97],
        [False, True, False, 0.03],

        [False, False, True, 0.001],
        [False, False, False, 0.999],
    ],
    [newbornBlindness, accidentBlindness])

guideDog = ConditionalProbabilityTable(
    [
        [True, True, True, 0.92],
        [True, True, False, 0.08],

        [True, False, True, 0.2],
        [True, False, False, 0.8],

        [False, True, True, 0.35],
        [False, True, False, 0.65],

        [False, False, True, 0.09],
        [False, False, False, 0.91],
    ],
    [labrador, blindMan])

s1 = State(location, name="Am I in the UK?")
s2 = State(weather, name="The weather is nice")
s3 = State(labrador, name="Meet a labrador")
s4 = State(pet, name="Labrador is a pet")
s5 = State(guideDog, name="Labrador is a guide dog")
s6 = State(newbornBlindness, name="Blindness in newborns")
s7 = State(accidentBlindness, name="Blindness by an accident")
s8 = State(blindMan, name="Meet a blind man")

model = BayesianNetwork()
model.add_states(s1, s2, s3, s4, s5, s6, s7, s8)
model.add_edge(s1, s3)
model.add_edge(s2, s3)
model.add_edge(s3, s4)
model.add_edge(s3, s5)
model.add_edge(s6, s8)
model.add_edge(s7, s8)
model.add_edge(s8, s5)
model.bake()

print(model.probability([False, False, True, False, True, False, False, False]))
print(model.probability([True, False, True, False, False, False, False, False]))
print(model.probability([True, True, True, False, False, False, False, False]))
"print(model.predict_proba([None, None, None,None, None, None,None, None]))"
