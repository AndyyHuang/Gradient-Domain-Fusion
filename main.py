import numpy as np
from scipy.sparse.linalg import lsqr
from scipy.sparse import lil_matrix
from PIL import Image

"""
Sparse Matrix

[-1 1 0 0 0 ... ]  [v(0, 0)]         [ s(0, 1) - s(0, 0)]
                    v(0, 1)                  .
                      .                      .
                      .        =             .
                      .
[               ]  [v(m, n)]         [s(m, n) - s(m - 1, n)]
"""

def load_image(path):
    return np.array(Image.open(path)) / 255.0

def show_image(np_im, rgb=False):
    np_im = np_im * 255.0
    if rgb:
        pil_im = Image.fromarray(np_im.astype('uint8'), 'RGB')
    else:
        pil_im = Image.fromarray(np_im.astype('uint8'), 'L')

    pil_im.show()
    return

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
            if channel != None:
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
            if channel != None:
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

im = load_image("source/toy_problem.jpeg")
im_recon = reconstruct_image(im)
show_image(im_recon)

# Show the original and reconstructed images


