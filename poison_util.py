import argparse
import os
from PIL import Image
import numpy as np
import math
from xai_utils import load_image

TARGETS = [0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8]

# Returns an Image whose average value over all pixels is target accurate to epsilon   
def gamma_bin_search(img: Image, target: float = 0.5, epsilon: float = 0.0001, max_iter: int = 50) -> Image:
    if target < 0.0 or target > 1.0:
        raise ValueError('target must be between 0.0 and 1.0 inclusive.')
    
    high = 9.99
    low = 0.01
    gamma = 1.00
    img_np = np.array(img, dtype=float)
    img_np /= 255
    # Clone input in case loop is skipped
    candidate = np.array(img, dtype=float)
    candidate /= 255

    mean = np.mean(img_np)
    iterations = 0
    while abs(mean - target) > epsilon and iterations < max_iter:
        iterations += 1
        # Apply gamma and check mean brightness
        candidate = img_np ** (1 / gamma)
        mean = np.mean(candidate)

        if mean < target:
            # Too dark, increase gamma
            low = gamma
        else:
            # Too light, decrease gamma
            high = gamma
        gamma = 10 ** ((math.log10(high) + math.log10(low)) / 2)

    # Convert the candidate to an image
    img_np = (candidate * 255).astype('uint8')
    return Image.fromarray(img_np, mode='L')

# Permute through TARGETS and Procedurals to generate Masks
def generate_masks(p_dir: str, m_dir: str) -> None:
    for f in os.listdir(p_dir):
        f_img, f_name = load_image(os.path.join(p_dir, f), mode='L')
        if f_img is not None:
            for t in TARGETS:
                result = gamma_bin_search(f_img, target=t)
                suffix = int(t * 100)
                result.save(f'{os.path.join(m_dir, f_name)}_L{suffix:02d}.png', optimize=True)

# Permute through base images, noises, and masks
def permute_noises_masks(b_dir: str, n_dir: str, m_dir: str, out_dir: str, alpha: float) -> None:
    for base_file in os.listdir(b_dir):
        base_img, base_name = load_image(os.path.join(b_dir, base_file), 'RGB')
        if base_img is not None:
            for ptrb_file in os.listdir(n_dir):
                ptrb_img, ptrb_name = load_image(os.path.join(n_dir, ptrb_file), 'RGB')
                if ptrb_img is not None:
                    for mask_file in os.listdir(m_dir):
                        mask_img, mask_name = load_image(os.path.join(m_dir, mask_file), 'L')
                        if mask_img is not None:
                            # Convert to NumPy for compositing
                            base_img_arr = np.array(base_img).astype(float)
                            ptrb_img_arr = np.array(ptrb_img).astype(float)
                            mask_img_arr = np.array(mask_img).astype(float) / 255
                            mask_img_arr = mask_img_arr[:, :, None]

                            # Add masked noise to base and clamp to 0-255
                            comp_arr = base_img_arr + (ptrb_img_arr * mask_img_arr * alpha)
                            comp_arr = np.clip(comp_arr, 0, 255).astype(np.uint8)

                            # Save image
                            comp_img = Image.fromarray(comp_arr, mode='RGB')
                            comp_img.save(f'{out_dir}/{base_name}_{ptrb_name}_{mask_name}.png', optimize=True)


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--bases', default='./noise_data/bases', help='Folder containing base images')
    parser.add_argument('--procedurals', default='./noise_data/procedurals', help='Folder containing procedural noises')
    parser.add_argument('--noises', default='./noise_data/noises', help='Folder containing noisy images')
    parser.add_argument('--masks', default='./noise_data/masks', help='Folder containing masks')
    parser.add_argument('--alpha', default='0.15', help='Master opacity for noises added to images')
    parser.add_argument('--output', default='./noise_data/results', help='Folder containing output images')
    arg_list = parser.parse_args()

    # Error handling
    if not os.path.exists(arg_list.bases) or not os.path.isdir(arg_list.bases):
        raise FileNotFoundError(f'{arg_list.bases} not found or is not directory')
    if not os.path.exists(arg_list.procedurals) or not os.path.isdir(arg_list.procedurals):
        raise FileNotFoundError(f'{arg_list.procedurals} not found or is not directory')
    if not os.path.exists(arg_list.noises) or not os.path.isdir(arg_list.noises):
        raise FileNotFoundError(f'{arg_list.noises} not found or is not directory')
    try:
        alpha = float(arg_list.alpha)
    except ValueError:
        print('alpha must be between 0.0 and 1.0 inclusive')   
    if not 0 <= alpha <= 1:
        raise ValueError('alpha must be between 0.0 and 1.0 inclusive')
    
    # Create destination folder
    os.makedirs(arg_list.masks, exist_ok=True)
    os.makedirs(arg_list.output, exist_ok=True)
    
    generate_masks(arg_list.procedurals, arg_list.masks)
    permute_noises_masks(arg_list.bases, arg_list.noises, arg_list.masks, arg_list.output, alpha)

    print('Generation complete.')
