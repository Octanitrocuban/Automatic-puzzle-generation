# -*- coding: utf-8 -*-
"""
Module to create random puzzle from a given picture.
"""

import numpy as np
from tqdm import tqdm
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
		Plugs dots positions. Note that if their x and y value are 0, it
		means that there are no plug (boarder of the puzzle).

	"""
	n_head = 5
	# number of dots per segment of tile
	n_seg = n_head+3
	centers = np.meshgrid(range(n[0]), range(n[1]))
	centers = np.array([np.ravel(centers[1]), np.ravel(centers[0])]).T
	# kernels rotation
	# k1 : |1 2|	  k2 : | 1 |	 k3 : |  2 3  |
	#	   |0 3|		   |0 2|		  |1	 4|
	#					   | 3 |		  |0	 5|
	#									  |  7 6  |
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
	reshape = midpoints.shape[0]*4
	for i in range(len(conns)):
		masque = np.sum(midpoints == conns[i], axis=2) == 2
		bruit = np.random.uniform(0.17, 0.27)*(-1)**np.random.randint(0, 2)
		bruit = round(bruit, 5)
		temp_mids = np.reshape(midpoints, (reshape, 2))
		temp_conn = np.reshape(conneteurs, (reshape, n_head, 2))
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
				temp_conn[where_rm[0]] = np.array([x_att[::-1],
												   y_att[::-1]]).T

				temp_conn[where_rm[1]] = np.array([x_att, y_att]).T

			else:
				# lower
				x_att = np.cos(theta+np.pi*7/10)*rayon
				y_att = np.sin(theta+np.pi*7/10)*rayon
				x_att = (temp_mids[masque, 0]+x_att[:, np.newaxis])[:, 0]
				y_att = (temp_mids[masque, 1]+y_att[:, np.newaxis])[:, 0]
				temp_conn[where_rm[0]] = np.array([x_att, y_att]).T
				temp_conn[where_rm[1]] = np.array([x_att[::-1],
												   y_att[::-1]]).T

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
				temp_conn[where_rm[0]] = np.array([x_att[::-1],
												   y_att[::-1]]).T

				temp_conn[where_rm[1]] = np.array([x_att, y_att]).T

			else:
				# left
				x_att = np.cos(theta+np.pi/5)*rayon
				y_att = np.sin(theta+np.pi/5)*rayon
				x_att = (temp_mids[masque, 0]+x_att[:, np.newaxis])[:, 0]
				y_att = (temp_mids[masque, 1]+y_att[:, np.newaxis])[:, 0]
				temp_conn[where_rm[0]] = np.array([x_att, y_att]).T
				temp_conn[where_rm[1]] = np.array([x_att[::-1],
												   y_att[::-1]]).T

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

		shp = vector.shape
		u_id = np.arange(0, shp[0]-1)
		v_id = np.arange(1, shp[0])
		norms_u = (vector[u_id, 0]**2 +vector[u_id, 1]**2)**.5
		norms_v = (vector[v_id, 0]**2 +vector[v_id, 1]**2)**.5
		sc_prods = (vector[u_id, 0]*vector[v_id, 0]
				   +vector[u_id, 1]*vector[v_id, 1])

		angles = np.arccos(sc_prods/(norms_u*norms_v))

		sub = piece[1:-1][angles != 0]
		sub = np.append([piece[0]], sub, axis=0)
		sub = np.append(sub, [piece[-1]], axis=0)
		minimum.append(sub)

		if plot:
			plt.figure(figsize=(6, 6))
			plt.plot(piece[:, 0], piece[:, 1], 'ko-', ms=8)
			plt.plot(sub[:, 0], sub[:, 1], 'r.-')
			plt.axis('equal')
			plt.show()

	return minimum

def extract_mask_corner_in_tiles(tiles, corners):
	"""
	Function to compute which tile's dots are corners.

	Parameters
	----------
	tiles : list
		List of the tiles of the puzzle.
	corners : numpy.ndarray
		Position of the corners of the tiles.

	Returns
	-------
	masques : list
		list of numpy.ndarray. Boolean mask indicating which tile's dots are
		corners.

	"""
	masques = []
	for i, tile in enumerate(tiles):
		masq = tile == corners[i, :, np.newaxis]
		masq = masq[:, :, 0]&masq[:, :, 1]
		masq =  masq[0]|masq[1]|masq[2]|masq[3]
		masques.append(masq)

	return masques

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
		scaling[:, 0] = (scaling[:, 0]+0.5)/(n[1]) * (shape[1] -1)
		scaling[:, 1] = (scaling[:, 1]+0.5)/(n[0]) * (shape[0] -1)
		scaled.append(scaling)

	return scaled

def contour_tile(tile, grid_shape, factor):
	"""
	Function to found which pixels are cut by the bound of the tile.

	Parameters
	----------
	tile : numpy.ndarray
		Positions of the dots defining the lines delimiting the tile.
	grid_shape : tuple
		Size of the picture (in number of pixel).
	factor : float
		Factor to increase or decrease the number of dots creating during the
		interpolation part.

	Returns
	-------
	pixels : numpy.ndarray
		2 dimensional array filled with 0 and 1. The 1 indicates witch pixel
		are being part of the input tile.

	Note
	----
	This method doesn't relly on equation, because it actually make an
	approximation of the cuts.

	"""
	if len(grid_shape) == 2:
		pixels = np.zeros(grid_shape)
	elif len(grid_shape) == 3:
		pixels = np.zeros((grid_shape[0], grid_shape[1]))
	else:
		raise ValueError('')

	diff = tile[1:]-tile[:-1]
	alpha = np.linspace(0, 1, factor)[:, np.newaxis]
	interp_x = (tile[:-1, 0, np.newaxis]+(diff[:, 0]*alpha).T)
	interp_y = (tile[:-1, 1, np.newaxis]+(diff[:, 1]*alpha).T)
	interp_x = np.round(np.ravel(interp_x)).astype(int)
	interp_y = np.round(np.ravel(interp_y)).astype(int)
	
	pixels[interp_y, interp_x] = 1
	return pixels

def contour_tile_M(tile, grid, factor):
	"""
	Function to found which pixels are cut by the bound of the tile.

	Parameters
	----------
	tile : numpy.ndarray
		Positions of the dots defining the lines delimiting the tile.
	grid: numpy.ndarray
		Grid on which the contour will be add.
	factor : float
		Factor to increase or decrease the number of dots creating during the
		interpolation part.

	Returns
	-------
	grid : numpy.ndarray
		2 dimensional array filled with 0 and 1. The 1 indicates witch pixel
		are being part of the input tile.

	Note
	----
	This method doesn't relly on equation, because it actually make an
	approximation of the cuts.

	"""
	diff = tile[1:]-tile[:-1]
	alpha = np.linspace(0, 1, factor)[:, np.newaxis]
	interp_x = (tile[:-1, 0, np.newaxis]+(diff[:, 0]*alpha).T)
	interp_y = (tile[:-1, 1, np.newaxis]+(diff[:, 1]*alpha).T)
	interp_x = np.round(np.ravel(interp_x)).astype(int)
	interp_y = np.round(np.ravel(interp_y)).astype(int)
	grid[interp_y, interp_x] = 1
	return grid

def interior_tile(carte, positions):
	"""
	Function to found witch pixels are being part of a tile.

	Parameters
	----------
	carte : numpy.ndarray
		A 2 dimensions array to explore.
	positions : numpy.ndarray
		Starting position of the exploration: np.array([[xi, yi]]).

	Returns
	-------
	carte : numpy.ndarray
		2 dimensionals array where 1 values indicate the pixels that are
		beeing part of the tile.

	"""
	shape = carte.shape
	kernel = np.array([[[-1,  0]], [[ 0, -1]], [[ 0,  1]], [[ 1,  0]]])
	while len(positions) > 0:
		carte[positions[:, 0], positions[:, 1]] = True
		positions = positions+kernel
		shape = positions.shape
		positions = np.reshape(positions, (shape[0]*shape[1], 2))
		positions = positions[carte[positions[:, 0],
									positions[:, 1]] == False]

		positions = np.unique(positions, axis=0)

	return carte

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

def animated_fill(n, tiles, tiles_corner, image, method, factor=200, freq=2,
				  fond='dark'):
	"""
	Function to create images of the filling of the puzzle. This function will
	be way slower than animated_fill_multi. animated_fill_multi is design for
	a batch of pieces at time filling.

	Parameters
	----------
	n : tuple
		Number of tiles to generate (in height, in width).
	tiles : list
		List of the tiles of the puzzle.
	tiles_corner : numpy.ndarray
		Position of the corners of the tiles.
	image : numpy.ndarray
		Picture use for the puzzle creation.
	method : str
		Method to fill the puzzle.
	factor : float, optional
		Factor to increase or decrease the number of dots creating during the
		interpolation part. The default is 200.
	freq : int, optional
		Frequency of the figure printing. Lower this factor will be (with
		minimum = 1) higher figure print number there will be. The default
		is 2.
	fond : str, optional
		Color of the background. The default is 'dark'.

	Raises
	------
	NotImplemented
		Asking a background color which isn't implemented.
	NotImplemented
		Asking a solving method which isn't implemented.

	Returns
	-------
	None.

	"""
	reverse_image = np.copy(image)[::-1]
	masques = extract_mask_corner_in_tiles(tiles, tiles_corner)
	kernel = np.array([[1, 1], [1, -1], [-1, -1], [-1, 1]])*4
	tiles_center = np.round(np.mean(tiles_corner, axis=1), 0).astype(int)
	if fond == 'dark':
		vide = np.zeros(image.shape, dtype='uint8')
	elif fond == 'white':
		vide = np.zeros(image.shape, dtype='uint8')+255
	else:
		raise NotImplemented("Asked background color isn't implemented.")

	q = 0
	order = np.arange(len(tiles))
	img_shape = image.shape
	if method == 'random':
		np.random.shuffle(order)
	elif method == 'sorted':
		pass
	else:
		raise NotImplemented("Asked filling method isn't implemented.")

	for i in tqdm(order):
		contour = contour_tile(tiles[i], img_shape, factor)
		st_corns = (tiles[i][masques[i]]).astype(int)[:-1]+kernel
		st_corns = st_corns[:, ::-1]
		l_ran_h = np.arange(st_corns[0][0], st_corns[1][0]+1, 1)
		lh = len(l_ran_h)
		l_ran_v = np.arange(st_corns[0][1], st_corns[2][1]+1, 1)
		lv = len(l_ran_v)
		l_zer_h = np.zeros(lh, dtype=int)
		l_zer_v = np.zeros(lv, dtype=int)
		L1 = contour[l_ran_h, l_zer_h+st_corns[0][1]]
		L2 = contour[l_zer_v+st_corns[1][0], l_ran_v]
		L3 = contour[l_ran_h, l_zer_h+st_corns[3][1]]
		L4 = contour[l_zer_v+st_corns[0][0], l_ran_v]
		starter = np.zeros((1+lh*2+lv*2, 2), dtype=int)
		starter[0] = tiles_center[i, 1], tiles_center[i, 0]
		if 1 not in L3:
			starter[1:lh+1, 0] = l_ran_h
			starter[1:lh+1, 1] = st_corns[3][1]
		else:
			mask = np.where(L3 == 1)[0]
			starter[1] = st_corns[3]
			starter[2] = st_corns[2]
			starter[3] = l_ran_h[mask[0]-1], st_corns[3][1]
			starter[4] = l_ran_h[mask[-1]+1], st_corns[2][1]

		if 1 not in L2:
			starter[lh+1:lh+lv+1, 0] = st_corns[1][0]
			starter[lh+1:lh+lv+1, 1] = l_ran_v
		else:
			mask = np.where(L2 == 1)[0]
			starter[lh+1] = st_corns[1]
			starter[lh+2] = st_corns[2]
			starter[lh+3] = st_corns[1][0], l_ran_v[mask[0]-1]
			starter[lh+4] = st_corns[2][0], l_ran_v[mask[-1]+1]

		if 1 not in L1:
			starter[lh+lv+1:lh+lh+lv+1, 0] = l_ran_h
			starter[lh+lv+1:lh+lh+lv+1, 1] = st_corns[0][1]
		else:
			mask = np.where(L1 == 1)[0]
			starter[lh+lv+1] = st_corns[0]
			starter[lh+lv+2] = st_corns[1]
			starter[lh+lv+3] = l_ran_h[mask[0]-1], st_corns[0][1]
			starter[lh+lv+4] = l_ran_h[mask[-1]+1], st_corns[1][1]

		if 1 not in L4:
			starter[lh+lh+lv+1:, 0] = l_zer_v+st_corns[0][0]
			starter[lh+lh+lv+1:, 1] = l_ran_v
		else:
			mask = np.where(L4 == 1)[0]
			starter[lh+lh+lv+1] = st_corns[0]
			starter[lh+lh+lv+2] = st_corns[3]
			starter[lh+lh+lv+3] = st_corns[0][0], l_ran_v[mask[0]-1]
			starter[lh+lh+lv+3] = st_corns[3][0], l_ran_v[mask[-1]+1]

		starter = starter[(starter[:, 0] > 0)&(starter[:, 1] > 0)]
		pixels_tile = interior_tile(contour.astype(bool), starter)
		vide[pixels_tile] = reverse_image[pixels_tile]
		q += 1
		if (q%freq) == 0:
			plt.figure(figsize=(12, 12))
			plt.imshow(vide, origin='lower', interpolation='none')
			plt.axis('off')
			plt.show()

	if (q%freq) != 0:
		plt.figure(figsize=(12, 12))
		plt.imshow(vide, origin='lower', interpolation='none')
		plt.axis('off')
		plt.show()

def animated_fill_multi(n, tiles, tiles_corner, image, method, factor=200,
						freq=2, fond='dark'):
	"""
	Function to create images of the filling of the puzzle. This function will
	be way faster than animated_fill. animated_fill is design for a piec by
	piece filling.

	Parameters
	----------
	n : tuple
		Number of tiles to generate (in height, in width).
	tiles : list
		List of the tiles of the puzzle.
	tiles_corner : numpy.ndarray
		Position of the corners of the tiles.
	image : numpy.ndarray
		Picture use for the puzzle creation.
	method : str
		Method to fill the puzzle.
	factor : float, optional
		Factor to increase or decrease the number of dots creating during the
		interpolation part. The default is 200.
	freq : int, optional
		Frequency of the figure printing. Lower this factor will be (with
		minimum = 1) higher figure print number there will be. The default
		is 2.
	fond : str, optional
		Color of the background. The default is 'dark'.

	Raises
	------
	NotImplemented
		Asking a background color which isn't implemented.
	NotImplemented
		Asking a solving method which isn't implemented.

	Returns
	-------
	None.

	"""
	reverse_image = np.copy(image)[::-1]
	masques = extract_mask_corner_in_tiles(tiles, tiles_corner)
	kernel = np.array([[1, 1], [1, -1], [-1, -1], [-1, 1]])*4
	tiles_center = np.round(np.mean(tiles_corner, axis=1), 0).astype(int)
	mask_img = np.zeros((image.shape[0], image.shape[1]))
	if fond == 'dark':
		vide = np.zeros(image.shape, dtype='uint8')
	elif fond == 'white':
		vide = np.zeros(image.shape, dtype='uint8')+255
	else:
		raise NotImplemented("Asked background color isn't implemented.")

	order = np.arange(len(tiles))
	img_shape = image.shape
	num_up = len(tiles)/freq
	if num_up == int(num_up):
		num_up = int(num_up)
	else:
		num_up = int(num_up)+1

	n_freq = min([freq, len(tiles)])
	if method == 'random':
		np.random.shuffle(order)
	elif method == 'sorted':
		pass
	else:
		raise NotImplemented("Asked filling method isn't implemented.")

	for i in tqdm(range(num_up)):
		to_fill_id = order[i*n_freq:(i+1)*n_freq]
		conc_nds = np.zeros((1, 2), dtype=int)
		count = 0
		for j in to_fill_id:
			mask_img = contour_tile_M(tiles[j], mask_img, factor)
			st_corns = (tiles[j][masques[j]]).astype(int)[:-1]+kernel
			st_corns = st_corns[:, ::-1]
			l_ran_h = np.arange(st_corns[0][0], st_corns[1][0]+1, 1)
			lh = len(l_ran_h)
			l_ran_v = np.arange(st_corns[0][1], st_corns[2][1]+1, 1)
			lv = len(l_ran_v)
			l_zer_h = np.zeros(lh, dtype=int)
			l_zer_v = np.zeros(lv, dtype=int)
			L1 = mask_img[l_ran_h, l_zer_h+st_corns[0][1]]
			L2 = mask_img[l_zer_v+st_corns[1][0], l_ran_v]
			L3 = mask_img[l_ran_h, l_zer_h+st_corns[3][1]]
			L4 = mask_img[l_zer_v+st_corns[0][0], l_ran_v]
			starter = np.zeros((1+lh*2+lv*2, 2), dtype=int)
			starter[0] = tiles_center[j, 1], tiles_center[j, 0]

			if 1 not in L3:
				starter[1:lh+1, 0] = l_ran_h
				starter[1:lh+1, 1] = st_corns[3][1]
			else:
				mask = np.where(L3 == 1)[0]
				starter[1] = st_corns[3]
				starter[2] = st_corns[2]
				starter[3] = l_ran_h[mask[0]-1], st_corns[3][1]
				starter[4] = l_ran_h[mask[-1]+1], st_corns[2][1]

			if 1 not in L2:
				starter[lh+1:lh+lv+1, 0] = st_corns[1][0]
				starter[lh+1:lh+lv+1, 1] = l_ran_v
			else:
				mask = np.where(L2 == 1)[0]
				starter[lh+1] = st_corns[1]
				starter[lh+2] = st_corns[2]
				starter[lh+3] = st_corns[1][0], l_ran_v[mask[0]-1]
				starter[lh+4] = st_corns[2][0], l_ran_v[mask[-1]+1]

			if 1 not in L1:
				starter[lh+lv+1:lh+lh+lv+1, 0] = l_ran_h
				starter[lh+lv+1:lh+lh+lv+1, 1] = st_corns[0][1]
			else:
				mask = np.where(L1 == 1)[0]
				starter[lh+lv+1] = st_corns[0]
				starter[lh+lv+2] = st_corns[1]
				starter[lh+lv+3] = l_ran_h[mask[0]-1], st_corns[0][1]
				starter[lh+lv+4] = l_ran_h[mask[-1]+1], st_corns[1][1]

			if 1 not in L4:
				starter[lh+lh+lv+1:, 0] = l_zer_v+st_corns[0][0]
				starter[lh+lh+lv+1:, 1] = l_ran_v
			else:
				mask = np.where(L4 == 1)[0]
				starter[lh+lh+lv+1] = st_corns[0]
				starter[lh+lh+lv+2] = st_corns[3]
				starter[lh+lh+lv+3] = st_corns[0][0], l_ran_v[mask[0]-1]
				starter[lh+lh+lv+3] = st_corns[3][0], l_ran_v[mask[-1]+1]

			starter = starter[(starter[:, 0] > 0)&(starter[:, 1] > 0)]
			conc_nds = np.concatenate((conc_nds, starter))

		pixels_tile = interior_tile(mask_img.astype(bool), conc_nds[1:])
		vide[pixels_tile] = reverse_image[pixels_tile]

		plt.figure(figsize=(12, 12))
		plt.imshow(vide, origin='lower', interpolation='none')
		plt.axis('off')
		plt.show()
