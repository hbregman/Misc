# -*- coding: utf-8 -*-
"""
Created on Sat Jan 16 10:13:35 2016

@author: halliebregman
"""

x= range(len(binder2830))
y1 = data1.y1
y2 = data1.y2
y3 = data1.y3

fig = plt.figure()
ax1 = fig.add_subplot(111)
ax1.plot(x, y1)
ax1.set_ylabel('y1')
ax1.set_ylim([0,1.1])

ax2 = ax1.twinx()
ax2.plot(x, y2, 'r-')
ax2.set_ylabel('y2', color='r')
for tl in ax2.get_yticklabels():
    tl.set_color('r') 
ax2.set_ylim([0, 25]) 

ax3 = ax1.twinx()
ax3.plot(x, y3, 'g-')
ax3.set_ylabel('y3', color='g')
for tl in ax3.get_yticklabels():
    tl.set_color('g')  
ax3.set_ylim([990, 1040]) 
