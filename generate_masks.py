import os
import argparse
import math
import numpy as np
from PIL import Image


def load_image(img_path: str) -> tuple[Image, str]:
    extensions = {'.jpg', '.jpeg', '.png', '.JPG', '.JPEG', '.PNG'}
    if os.path.splitext(img_path)[1] in extensions:
        name = os.path.splitext(os.path.basename(img_path))[0]
        try:
            img = Image.open(img_path)
            if img.mode == 'I;16':
                bit16 = np.array(img, dtype=np.uint16)
                bit8 = (bit16 / 256).astype('uint8')
                img = Image.fromarray(bit8, mode='L')
            else:
                img = img.convert('L')
            return img, name
        except Exception as e:
            print(f'Error loading {img_path}')
            print(e)
            return None, None
    else:
        return None, None


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


if __name__ == '__main__':
    targets = [0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8]
    parser = argparse.ArgumentParser()
    parser.add_argument('--folder', default='./noise_data/procedurals', help='The directory containing base procedural noises')
    parser.add_argument('--output', default='./noise_data/masks', help='The directory to save to')
    arg_list = parser.parse_args()

    # Error handling
    if not os.path.exists(arg_list.folder) or not os.path.isdir(arg_list.folder):
        raise FileNotFoundError(f'{arg_list.folder} not found or is not directory')
    
    # Create destination folder
    os.makedirs(arg_list.output, exist_ok=True)

    for f in os.listdir(arg_list.folder):
        f_img, f_name = load_image(os.path.join(arg_list.folder, f))
        if f_img is not None:
            for t in targets:
                result = gamma_bin_search(f_img, target=t)
                suffix = int(t * 100)
                result.save(f'{os.path.join(arg_list.output, f_name)}_L{suffix:02d}.png')