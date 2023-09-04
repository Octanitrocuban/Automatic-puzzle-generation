# -*- coding: utf-8 -*-
"""
Created on Sat Sep  2 14:01:00 2023

@author: Matthieu Nougaret
"""
import creation
import matplotlib.pyplot as plt

n = (20, 31)
root = '../pictures/'
name = 'name'
img = plt.imread(root+name+'.jpg')

tiles, cents, coins, atts, midpts, conns = creation.puzzle_net(n)
tiles = creation.minimize_tiles(tiles)
scaled = creation.scale_tiles(n, tiles, img)
creation.show_puzzle(scaled, img, lw=2, figsize=(30, 30),
					 save_path=root+name+'-puzzle'+'.png',
					 color='limegreen')

creation.animated_fill(n, scaled, coins, img, 'random', freq=10)
