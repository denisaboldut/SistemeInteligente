# -*- coding: utf-8 -*-
"""
Created on Thu Mar  5 20:18:59 2020

@author: boldutdenisa
"""

"""
--------meeting a labrador ------------
most Labrador Retriver registrations are in the UK (18 554)- most likely place to meet one
    1/44 (contries in Europe) = 0.022
the weather is nice- most likely to meet one outside
    3/7 (3 days per week are sunny) = 0.428
    
--------Labrador----------------------
labradors can be:
  - pet
  - guide dog 
  - seizure dog
  
-------meeting a blind man---------------
1 of 1000 children are born blind
   1/1000 = 0.001
1 of 800 people suffering from blindness caused by accident 
   1/800 = 0.00125 
 
 => meeting a blind man 
       => increase the probability of meeting a guide dog   
 ----------------------------------------------
 
 ------meeting a children with epilepsy------

About 5.48 million people are estimated to suffer from severe traumatic brain injury (TBI) each year (73 cases per 100,000 people)
    73/100 000 = 0.00073

About 1.6 percent of women face difficulties at giving birth each year
   1.6/100 = 0.016
   
 => meeting a children with epilepsy
         => increase the probability of meeting a seizure dog  
 -------------------------------------------
 
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

headInjury = DiscreteDistribution({True: 0.0073, False: 0.9927})
difficultyBirth = DiscreteDistribution({True: 0.016, False: 0.984})
childrenEpilepsy = ConditionalProbabilityTable(
    [
        [True, True, True, 0.83],
        [True, True, False, 0.17],

        [True, False, True, 0.72],
        [True, False, False, 0.28],

        [False, True, True, 0.35],
        [False, True, False, 0.65],

        [False, False, True, 0.21],
        [False, False, False, 0.79],

    ],
    [headInjury, difficultyBirth]
)

seizureDog = ConditionalProbabilityTable(

    [
        [True, True, True, 0.79],
        [True, True, False, 0.21],

        [True, False, True, 0.32],
        [True, False, False, 0.68],

        [False, True, True, 0.11],
        [False, True, False, 0.89],

        [False, False, True, 0.03],
        [False, False, False, 0.97],



    ],
    [childrenEpilepsy, labrador]
)


s1 = State(location, name="Am I in the UK?")
s2 = State(weather, name="The weather is nice")
s3 = State(labrador, name="Meet a labrador")
s4 = State(pet, name="Labrador is a pet")
s5 = State(guideDog, name="Labrador is a guide dog")
s6 = State(newbornBlindness, name="Blindness in newborns")
s7 = State(accidentBlindness, name="Blindness by an accident")
s8 = State(blindMan, name="Meet a blind man")
s9 = State(headInjury, name="Severe head injury")
s10 = State(difficultyBirth, name="Difficulties ath birth")
s11 = State(childrenEpilepsy, name="Meet a children with epilepsy")
s12 = State(seizureDog, name="Seizure dog")

model = BayesianNetwork()
model.add_states(s1, s2, s3, s4, s5, s6, s7, s8, s9, s10, s11, s12)
model.add_edge(s1, s3)
model.add_edge(s2, s3)
model.add_edge(s3, s4)
model.add_edge(s3, s5)
model.add_edge(s6, s8)
model.add_edge(s7, s8)
model.add_edge(s8, s5)
model.add_edge(s9, s11)
model.add_edge(s10, s11)
model.add_edge(s11, s12)
model.add_edge(s3, s12)
model.bake()

print(model.probability([False, False, True, False, True, False, False, False, False, False, False, True]))
print(model.probability([True, False, True, False, False, False, False, False, False, False, False, False]))
print(model.probability([True, True, True, False, False, False, False, False, False, False, False, False]))
print(model.predict_proba([None, None, None,None, None, None,None, None,  None, None, None, None]))
