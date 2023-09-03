# -*- coding: utf-8 -*-
"""
Created on Sat Sep  2 14:01:00 2023

@author: Matthieu Nougaret
"""
import creation
import matplotlib.pyplot as plt

n = (37, 56)
img = plt.imread('../pictures/Volcan-Llaima-y-Laguna-Conguillio-desde-Sierra-Nevada.jpg')
#('../pictures/fontainebleau-forest.jpg')

tiles, cents, coins, atts, midpts, conns = creation.puzzle_net(n)
tiles = creation.minimize_tiles(tiles)
scaled = creation.scale_tiles(n, tiles, img)
# creation.show_puzzle(scaled, img, lw=0.75)
creation.animated_fill(n, scaled, coins, img, 'random', freq=21)
