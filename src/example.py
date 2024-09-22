# -*- coding: utf-8 -*-

import time
import creation
import numpy as np
from tqdm import tqdm

# import an image
root = '../pictures/'
name = 'Volcan-Llaima-y-Laguna-Conguillio-desde-Sierra-Nevada'
img = plt.imread(root+name+'.jpg')

# compute the number of pieces for each axis
n = list(img.shape)[:-1]
n[0] = int(round(n[0])/100)
n[1] = int(round(n[1])/100)

# compute the network of the puzzle
tiles, cents, coins, atts, midpts, conns = creation.puzzle_net(n)

# compute a simplified version of the tiles. Its aim is to remove useless dot
# to speed computation and minimize memory consumption
tiles = creation.minimize_tiles(tiles, False)

# scaled (up) the tiles arrays to match the image shape
scaled = creation.scale_tiles(n, tiles, img)

# show and save the image with the tile network on it
creation.show_puzzle(scaled, img, lw=2, figsize=(30, 30),
					 save_path=root+name+'-puzzle2'+'.png',
					 color='red')

# compute the scaled (up) corners position
coins_sc = creation.scale_tiles(n, coins, img)
coins_sc = np.array(coins_sc)

# create an animation on the filling of the puzzle. Optimized for a tile per
# tile filling.
creation.animated_fill(n, scaled, coins_sc, img, 'random', factor=105,
					   freq=100)

# create an animation on the filling of the puzzle. Optimized for a per batch
# tiles filling.
creation.animated_fill(n, scaled, coins_sc, img, 'sorted', factor=105,
						freq=100)

