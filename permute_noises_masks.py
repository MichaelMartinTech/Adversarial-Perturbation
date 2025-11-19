import argparse
import os
from PIL import Image
import numpy as np


def load_image(img_path: str, mode: str) -> tuple[Image, str]:
    extensions = {'.jpg', '.jpeg', '.png', '.JPG', '.JPEG', '.PNG'}
    if os.path.splitext(img_path)[1] in extensions:
        name = os.path.splitext(os.path.basename(img_path))[0]
        try:
            img = Image.open(img_path)
            if mode == 'RGB':
                img = img.convert('RGB')
            elif mode == 'L':
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


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--image', required=True, help='Clean base image')
    parser.add_argument('--noises', default='./noise_data/noises', help='Folder containing noisy images')
    parser.add_argument('--masks', default='./noise_data/masks', help='Folder containing masks')
    parser.add_argument('--output', default='./noise_data/results', help='Folder containing output images')
    arg_list = parser.parse_args()

    # Error handling
    if not os.path.exists(arg_list.image):
        raise FileNotFoundError(f'{arg_list.image} not found')
    if not os.path.exists(arg_list.noises) or not os.path.isdir(arg_list.noises):
        raise FileNotFoundError(f'{arg_list.perturbed} not found or is not directory')
    if not os.path.exists(arg_list.masks) or not os.path.isdir(arg_list.masks):
        raise FileNotFoundError(f'{arg_list.masks} not found or is not directory')
    
    # Create destination folder
    os.makedirs(arg_list.output, exist_ok=True)
    
    # Load and composite images
    base_img, base_name = load_image(arg_list.image, 'RGB')

    for ptrb_file in os.listdir(arg_list.noises):
        ptrb_img, ptrb_name = load_image(os.path.join(arg_list.noises, ptrb_file), 'RGB')
        if ptrb_img is not None:
            for mask_file in os.listdir(arg_list.masks):
                mask_img, mask_name = load_image(os.path.join(arg_list.masks, mask_file), 'L')
                if mask_img is not None:
                    # Convert to NumPy for compositing
                    base_img_arr = np.array(base_img).astype(float)
                    ptrb_img_arr = np.array(ptrb_img).astype(float)
                    mask_img_arr = np.array(mask_img).astype(float) / 255
                    mask_img_arr = mask_img_arr[:, :, None]

                    # Add masked noise to base and clamp to 0-255
                    comp_arr = base_img_arr + (ptrb_img_arr * mask_img_arr * 0.15)
                    comp_arr = np.clip(comp_arr, 0, 255).astype(np.uint8)

                    # Save image
                    comp_img = Image.fromarray(comp_arr, mode='RGB')
                    comp_img.save(f'{arg_list.output}/{base_name}_{ptrb_name}_{mask_name}.png')
   