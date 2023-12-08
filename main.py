from utils import *

"""
Sparse Matrix

[-1 1 0 0 0 ... ]  [v(0, 0)]         [ s(0, 1) - s(0, 0)]
                    v(0, 1)                  .
                      .                      .
                      .        =             .
                      .
[               ]  [v(m, n)]         [s(m, n) - s(m - 1, n)]
"""

source_im = load_image("source/spiral_galaxy.jpg")
target_im = load_image("target/nightsky.jpg")
mask_im = load_mask("masks/galaxy_mask.jpg")

new_im, coords_row, coords_col = apply_mask(source_im, target_im, mask_im)
coords = np.hstack([coords_row[:, None], coords_col[:, None]])

show_image(new_im, save_path='output/unblended_galaxy.jpg')

r, g, b = blend(source_im, target_im, coords, mask_im, 0), blend(source_im, target_im, coords, mask_im, 1), blend(source_im, target_im, coords, mask_im, 2)
recon_source = np.dstack([r, g, b]).clip(0, 1)
blended_im = np.array(target_im)
blended_im[coords_row, coords_col] = recon_source
show_image(blended_im, save_path='output/blended_galaxy.jpg')
