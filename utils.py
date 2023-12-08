import numpy as np
from scipy.sparse.linalg import lsqr
from scipy.sparse import lil_matrix
from PIL import Image
import matplotlib.pyplot as plt
from utils import *

def load_image(path, rgb=True):
    if rgb:
        return np.array(Image.open(path)) / 255.0
    else:
        return np.array(Image.open(path).convert('L')) / 255.0

def load_mask(path):
    return ((np.array(Image.open(path).convert('L')) / 255.0) > 0.5).astype(float)[:, :, None]

def show_image(np_im, rgb=True, save_path=None):
    np_im = np_im * 255.0
    if rgb:
        pil_im = Image.fromarray(np_im.astype('uint8'), 'RGB')
    else:
        pil_im = Image.fromarray(np_im.astype('uint8'), 'L')

    if save_path != None:
        pil_im.save(save_path)
    pil_im.show()
    return

def apply_mask(source_im, target_im, mask_im):
    new_im = mask_im * source_im + (1 - mask_im) * target_im
    coords_row, coords_col = np.where(np.all(mask_im > [[0.5, 0.5, 0.5]], axis=-1))
    return new_im, coords_row, coords_col

# 2.1 Toy Problem
def reconstruct_image(src_im, channel=None):
    """Reconstruct the image v from its gradients and a single pixel intensity."""
    # Compute gradients of s
    rows, cols = src_im.shape[0], src_im.shape[1]
    num_eqs = (rows - 1) * cols + rows * (cols - 1) + 1

    # Matrix A (sparse matrix with -1 and 1's)
    A = lil_matrix((num_eqs, rows * cols))
    b = np.zeros(num_eqs)

    # Indexes
    im2var = np.arange(rows * cols).reshape(rows, cols)

    eq = 0
    # X gradients
    for y in range(rows):
        for x in range(cols - 1):
            A[eq, im2var[y, x + 1]] = 1
            A[eq, im2var[y, x]] = -1

            # Source x gradient
            if channel == None:
                b[eq] = src_im[y, x + 1] - src_im[y, x]
            else:
                b[eq] = src_im[y, x + 1][channel] - src_im[y, x][channel]
            eq += 1

    # Y gradients
    for y in range(rows - 1):
        for x in range(cols):
            A[eq, im2var[y + 1, x]] = 1
            A[eq, im2var[y, x]] = -1

            # Source y gradient
            if channel == None:
                b[eq] = src_im[y + 1, x] - src_im[y, x]
            else:
                b[eq] = src_im[y + 1, x][channel] - src_im[y, x][channel]
            eq += 1

    # Top left color constant
    A[eq, im2var[0, 0]] = 1

    if channel != None:
        b[eq] = src_im[0, 0][channel]
    else:
        b[eq] = src_im[0, 0]

    # Solve the least squares problem
    A = A.tocsr()
    sol = lsqr(A, b)[0]

    # Reshape the solution to the image shape
    return sol.reshape(rows, cols)

def blend(source_im, target_im, coords, mask, channel):
    rows, cols = target_im.shape[0], target_im.shape[1]
    num_eqs = len(coords) * 4 + 1

    # Matrix A (sparse matrix with -1 and 1's)
    A = lil_matrix((num_eqs, len(coords)))
    b = np.zeros(num_eqs)

    im2var = np.zeros((rows, cols))
    im2var[coords[:, 0], coords[:, 1]] = np.arange(len(coords))

    eq = 0
    for coord in coords:
        row, col = coord[0], coord[1]
        neighbors = [(row, col + 1), (row, col - 1), (row + 1, col), (row - 1, col)]
        for y, x in neighbors:
            # If neighbor is falls in black of mask, it is target
            if (mask[y, x] < 0.5)[0]:
                b[eq] += target_im[y, x, channel]
            else:
                A[eq, im2var[y, x]] = -1
            A[eq, im2var[row, col]] = 1

            b[eq] += source_im[row, col, channel] - source_im[y, x, channel]

            eq += 1

    # Top left color constant
    A[eq, im2var[coords[0][0], coords[0][1]]] = 1
    b[eq] = source_im[coords[0][0], coords[0][1], channel]

    # Solve the least squares problem
    A = A.tocsr()
    sol = lsqr(A, b)[0]
    return sol
