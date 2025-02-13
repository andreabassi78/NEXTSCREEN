# -*- coding: utf-8 -*-
"""
Created on Sat Mar  6 11:27:55 2021

@author: Andrea
"""

from matplotlib import pyplot as plt

plt.style.use('fivethirtyeight')
plt.rc('font', family='serif', size=10)


slices = [29,45,15,22,41]
labels = ['first','second','third','fourth','fifth']
explode = [0,0,0,0.1,0]


fig = plt.figure(figsize=(4,4), dpi=300)


fig.set_facecolor('w')
ax = fig.add_subplot(111)


ax.pie(slices, labels = labels, explode = explode, 
        shadow = True,
        startangle = 90,
        autopct = '%1.1f%%', #add percentage with one decimal
        wedgeprops = {'edgecolor': 'black'})

ax.set_title('My pie graph')

plt.rcParams.update(plt.rcParamsDefault)
plt.show()