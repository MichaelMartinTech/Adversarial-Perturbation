import argparse
import os
from PIL import Image
import numpy as np


def load_image(img_path, mode):
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
            return None


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--image', required=True, help='Clean base image')
    parser.add_argument('--perturbed', default='./noise_data/perturbed', help='Folder containing perturbed images')
    parser.add_argument('--masks', default='./noise_data/masks', help='Folder containing masks')
    parser.add_argument('--output', default='./noise_data/results', help='Folder containing output images')
    arg_list = parser.parse_args()

    # Error handling
    if not os.path.exists(arg_list.image):
        raise FileNotFoundError(f'{arg_list.image} not found')
    if not os.path.exists(arg_list.perturbed) or not os.path.isdir(arg_list.perturbed):
        raise FileNotFoundError(f'{arg_list.perturbed} not found or is not directory')
    if not os.path.exists(arg_list.masks) or not os.path.isdir(arg_list.masks):
        raise FileNotFoundError(f'{arg_list.masks} not found or is not directory')
    
    # Create destination folder
    os.makedirs(arg_list.output, exist_ok=True)
    
    # Load and composite images
    base_img, base_name = load_image(arg_list.image, 'RGB')

    for ptrb_file in os.listdir(arg_list.perturbed):
        ptrb_img, ptrb_name = load_image(os.path.join(arg_list.perturbed, ptrb_file), 'RGB')
        if ptrb_img is not None:
            for mask_file in os.listdir(arg_list.masks):
                mask_img, mask_name = load_image(os.path.join(arg_list.masks, mask_file), 'L')
                if mask_img is not None:
                    comp = Image.composite(ptrb_img, base_img, mask_img)
                    comp.save(f'{arg_list.output}/{base_name}_{ptrb_name}_{mask_name}.png')
   