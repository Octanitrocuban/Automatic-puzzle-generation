# -*- coding: utf-8 -*-
"""
Created on Tue Aug 29 22:06:18 2023

@author: Matthieu Nougaret
"""

import numpy as np
import matplotlib.pyplot as plt

def puzzle_net(n):
	"""
	Function to create the tiles of the puzzle.

	Parameters
	----------
	n : tuple
		Number of tiles to generate (in height, in width).

	Returns
	-------
	tiles : list
		List of the tiles of the puzzle. Note that there are usless positions
		coming from the creation method. These dots can be removed with the
		function minimize_tiles().
	centers : numpy.ndarray
		Position of the centers of the tiles.
	coins : numpy.ndarray
		Position of the corners of the tiles.
	attaches : numpy.ndarray
		Position of the start of the neck to the plugs of the tiles.
	midpoints : numpy.ndarray
		Position of the midle distance between two corners.
	conneteurs : numpy.ndarray
		Plugs dots positions. Note that if their x and y value are 0, it means
		that there are no plug (boarder of the puzzle).

	"""
	n_head = 5
	# number of dots per segment of tile
	n_seg = n_head+3
	centers = np.meshgrid(range(n[0]), range(n[1]))
	centers = np.array([np.ravel(centers[1]), np.ravel(centers[0])]).T
	# kernels rotation
	# k1 : |1 2|      k2 : | 1 |     k3 : |  2 3  |
	#      |0 3|           |0 2|          |1     4|
	#                      | 3 |          |0     5|
	#                                     |  7 6  |
	k1 = np.array([[-.5, -.5], [-.5, .5], [.5, .5], [.5, -.5]])
	k2 = np.array([[-.5, 0], [0, .5], [.5, 0], [0, -.5]])
	k3 = np.array([[-.5, -.075], [-.5, .075], [-.075, .5], [.075, .5],
				   [.5, .075], [.5, -.075], [.075, -.5], [-.075, -.5]])

	coins = centers+k1[:, np.newaxis]
	coins = np.transpose(coins, (1, 0, 2))
	midpoints = centers+k2[:, np.newaxis]
	midpoints = np.transpose(midpoints, (1, 0, 2))
	attaches = centers+k3[:, np.newaxis]
	attaches = np.transpose(attaches, (1, 0, 2))
	conneteurs = np.zeros((n[0]*n[1], 4, n_head, 2))
	uniques, comptes = np.unique(np.concatenate(midpoints),
								 axis=0, return_counts=True)

	keys_cent_4 = np.arange(4*n[0]*n[1]).reshape((n[0]*n[1], 4))
	conns = uniques[comptes > 1]
	theta = np.deg2rad(np.arange(0, 360, 360/n_head))
	for i in range(len(conns)):
		masque = np.sum(midpoints == conns[i], axis=2) == 2
		bruit = np.random.uniform(0.17, 0.27)*(-1)**np.random.randint(0, 2)
		bruit = round(bruit, 5)
		temp_mids = np.concatenate(midpoints)
		temp_conn = np.concatenate(conneteurs)
		rayon = round(np.random.uniform(0.08, 0.14), 3)
		argw = np.copy(np.argwhere(masque))
		if 1 in argw[:, 1]:
			masque = np.ravel(masque)
			where_rm = np.where(masque)[0]
			# upper / lower
			temp_mids[masque, 1] += bruit
			# add circle / some dots around
			if bruit > 0:
				# upper
				x_att = np.cos(theta-np.pi*3/10)*rayon
				y_att = np.sin(theta-np.pi*3/10)*rayon
				x_att = (temp_mids[masque, 0]+x_att[:, np.newaxis])[:, 0]
				y_att = (temp_mids[masque, 1]+y_att[:, np.newaxis])[:, 0]
				temp_conn[where_rm[0]] = np.array([x_att[::-1], y_att[::-1]]).T
				temp_conn[where_rm[1]] = np.array([x_att, y_att]).T

			else:
				# lower
				x_att = np.cos(theta+np.pi*7/10)*rayon
				y_att = np.sin(theta+np.pi*7/10)*rayon
				x_att = (temp_mids[masque, 0]+x_att[:, np.newaxis])[:, 0]
				y_att = (temp_mids[masque, 1]+y_att[:, np.newaxis])[:, 0]
				temp_conn[where_rm[0]] = np.array([x_att, y_att]).T
				temp_conn[where_rm[1]] = np.array([x_att[::-1], y_att[::-1]]).T

		else:
			masque = np.ravel(masque)
			where_rm = np.where(masque)[0]
			# right / left
			temp_mids[masque, 0] += bruit
			# add circle / some dots around
			if bruit > 0:
				# right
				x_att = np.cos(theta+np.pi*12/10)*rayon
				y_att = np.sin(theta+np.pi*12/10)*rayon
				x_att = (temp_mids[masque, 0]+x_att[:, np.newaxis])[:, 0]
				y_att = (temp_mids[masque, 1]+y_att[:, np.newaxis])[:, 0]
				temp_conn[where_rm[0]] = np.array([x_att[::-1], y_att[::-1]]).T
				temp_conn[where_rm[1]] = np.array([x_att, y_att]).T

			else:
				# left
				x_att = np.cos(theta+np.pi/5)*rayon
				y_att = np.sin(theta+np.pi/5)*rayon
				x_att = (temp_mids[masque, 0]+x_att[:, np.newaxis])[:, 0]
				y_att = (temp_mids[masque, 1]+y_att[:, np.newaxis])[:, 0]
				temp_conn[where_rm[0]] = np.array([x_att, y_att]).T
				temp_conn[where_rm[1]] = np.array([x_att[::-1], y_att[::-1]]).T

		# interpolation + smooth it (to do)
		midpoints = np.reshape(temp_mids[np.newaxis], midpoints.shape)
		conneteurs = np.reshape(temp_conn[np.newaxis], conneteurs.shape)

	conneteurs = np.concatenate(conneteurs)
	conneteurs = np.copy(conneteurs).reshape((n[0]*n[1], 4, 5, 2))

	# stacking coins, attaches & conneteurs to create the tiles
	tiles = np.zeros((n[0]*n[1], 4*n_seg+1, 2))
	tiles[:, :-1:n_seg] = coins
	tiles[:, 1::n_seg] = attaches[:, ::2]
	tiles[:, 7::n_seg] = attaches[:, 1::2]
	tiles[:, -1] = coins[:, 0]
	tiles = list(tiles)
	for i in range(n[0]*n[1]):
		tiles[i][2:2+n_head] = conneteurs[i, 0]
		tiles[i][5+n_head:5+n_head*2] = conneteurs[i, 1]
		tiles[i][8+n_head*2:8+n_head*3] = conneteurs[i, 2]
		tiles[i][11+n_head*3:11+n_head*4] = conneteurs[i, 3]
		tiles[i] = tiles[i][np.sum(tiles[i] == 0, axis=1) == 0]

	return tiles, centers, coins, attaches, midpoints, conneteurs

def minimize_tiles(tiles, plot=False):
	"""
	Function to remove useless dots from the tiles. The i-dot is considered
	as use less if the angle between the vectors (i-1 to i) & (i to i+1) have
	an angle equal to 0.

	Parameters
	----------
	tiles : list
		List of the tiles of the puzzle. Note that there are usless positions
		coming from the creation method.
	plot : bool, optional
		If you want to be plot the older & new tiles to check if there are
		errors. The default is False.

	Returns
	-------
	minimum : list
		List of the tiles of the puzzle without usless positions.

	"""
	minimum = []
	for i, piece in enumerate(tiles):
		vector = np.array([piece[1:, 1]-piece[:-1, 1],
						   piece[1:, 0]-piece[:-1, 0]]).T
		# initialisation
		u = vector[0]
		min_tile = []
		min_tile.append(piece[0])
		for j in range(len(vector)):
			v = vector[j]
			# scalar product
			sc_prod = vector[j, 0]*u[0]+vector[j, 1]*u[1]
			# vector normes
			norm_u = (u[0]**2 + u[1]**2)**.5
			norm_v = (v[0]**2 + v[1]**2)**.5
			angles = np.arccos(sc_prod/(norm_u*norm_v))
			if angles == 0:
				u = u+v
			else:
				u = v
				min_tile.append(piece[j])

		# to keep the tiles in a closed path
		min_tile.append(piece[-1])
		min_tile = np.array(min_tile)
		minimum.append(min_tile)

		if plot:
			plt.figure(figsize=(6, 6))
			plt.plot(piece[:, 0], piece[:, 1], 'ko-')
			plt.plot(min_tile[:, 0], min_tile[:, 1], 'r.-')
			plt.axis('equal')
			plt.show()

	return minimum

def scale_tiles(n, tiles, image):
	"""
	Function to scale the tiles to the size of the picture that will be
	transformed in puzzle. Note that given the number of tiles asked and the
	shape of the picture, the scaled tiles can be distorted (elongated /
	applative).

	Parameters
	----------
	tiles : list
		List of the tiles of the puzzle without usless positions.
	image : numpy.ndarray
		Picture use for the puzzle creation.

	Returns
	-------
	scaled : list
		List of the tiles of the puzzle without usless positions and scaled
		to fit the size of the wanted picture.

	"""
	shape = image.shape
	scaled = []
	for i, piece in enumerate(tiles):
		scaling = np.copy(piece)
		scaling[:, 0] = (scaling[:, 0]+0.5)/(n[1]) * shape[1]
		scaling[:, 1] = (scaling[:, 1]+0.5)/(n[0]) * shape[0]
		scaled.append(scaling)

	return scaled

def plot_tiling(n, tiles=None, coins=None, attaches=None, midpoints=None,
				centers=None, figsize=(15, 15)):
	"""
	Function to plot different part of the tiles.

	Parameters
	----------
	n : tuple
		Number of tiles to generate (in height, in width).
	tiles : list, optional
		List of the tiles of the puzzle. The default is None.
	coins : numpy.ndarray, optional
		Position of the corners of the tiles. The default is None.
	attaches : numpy.ndarray, optional
		Position of the start of the neck to the plugs of the tiles. The
		default is None.
	midpoints : numpy.ndarray, optional
		Position of the midle distance between two corners. The default is
		None.
	centers : numpy.ndarray, optional
		Position of the centers of the tiles. The default is None.
	figsize : tuple, optional
		Size of the figure. The default is (15, 15).

	Returns
	-------
	None.

	"""

	plt.figure(figsize=figsize)
	labeling = True
	for i in range(n[0]*n[1]):
		if labeling:
			labeling = False
			if type(tiles) != type(None):
				plt.plot(tiles[i][:, 0], tiles[i][:, 1], 'k',
						 label='pourtour')

			if type(coins) != type(None):
				plt.plot(coins[i, :, 0], coins[i, :, 1], 'ro', label='coins')

			if type(attaches) != type(None):
				plt.plot(attaches[i, :, 0], attaches[i, :, 1], 'gs',
						 label='neck')

			if type(midpoints) != type(None):
				plt.plot(midpoints[i, :, 0], midpoints[i, :, 1], 'bo',
						 label='attaches')

		else:
			if type(tiles) != type(None):
				plt.plot(tiles[i][:, 0], tiles[i][:, 1], 'k')

			if type(coins) != type(None):
				plt.plot(coins[i, :, 0], coins[i, :, 1], 'ro')

			if type(attaches) != type(None):
				plt.plot(attaches[i, :, 0], attaches[i, :, 1], 'gs')

			if type(midpoints) != type(None):
				plt.plot(midpoints[i, :, 0], midpoints[i, :, 1], 'bo')

	if type(centers) != type(None):
		plt.plot(centers[:, 0], centers[:, 1], 'k*', label='centre')

	plt.axis('equal')
	plt.xlim(-1, n[1])
	plt.ylim(-1, n[0])
	plt.legend(loc=[1.01, 0.6])
	plt.show()

def show_puzzle(tiles, image, figsize=(18, 18), lw=1, color='red',
				save_path=None):
	"""
	Function to show and save the representation of the puzzle.

	Parameters
	----------
	tiles : list
		List of the tiles of the puzzle. The default is None.
	image : numpy.ndarray
		Picture use for the puzzle creation.
	figsize : tuple, optional
		Size of the figure. The default is (18, 18).
	lw : float, optional
		Thickness of the lines representing the tiles of the puzzle. The
		default is 1.
	color : str or list, optional
		Color of the lines representing the tiles of the puzzle. The default
		is 'red'.
	save_path : str, optional
		Path for saving the figure. The default is None.

	Returns
	-------
	None.

	"""
	# making it up to down due to the way of ploting by plt.imshow.
	im_puzzle = np.copy(image)[::-1]
	shape = image.shape

	plt.figure(figsize=figsize)
	plt.imshow(im_puzzle, origin='lower', zorder=1, interpolation='none')
	for i, piece in enumerate(tiles):
		plt.plot(piece[:, 0], piece[:, 1], color=color, zorder=2, lw=lw)

	plt.xlim(-0.5, shape[1])
	plt.ylim(-0.5, shape[0])
	plt.axis('off')
	if type(save_path) == str:
		plt.savefig(save_path, bbox_inches='tight', transparent=True)

	plt.show()
