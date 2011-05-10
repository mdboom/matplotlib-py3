#!/usr/bin/env python

# override the default reporting of coords

from pylab import *

def millions(x):
    return '${:1.1f}M'.format(x*1e-6)

x =     rand(20)
y =     1e7*rand(20)

ax = subplot(111)
ax.fmt_ydata = millions
plot(x, y, 'o')

show()

