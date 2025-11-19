import torch
from enum import Enum
import matplotlib.patches as mpatches
from PIL import Image
import os
import numpy as np

extensions = {'.jpg', '.jpeg', '.png', '.JPG', '.JPEG', '.PNG'}

def load_image(img_path: str, mode: str) -> tuple[Image, str]: 
    if os.path.splitext(img_path)[1] in extensions:
        name = os.path.splitext(os.path.basename(img_path))[0]
        try:
            img = Image.open(img_path)
            if mode == 'RGB':
                img = img.convert('RGB')
            elif mode == 'L':
                # Convert 16-bit to 8-bit if needed
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

def get_device():
    if torch.cuda.is_available():
        return 'cuda'
    elif torch.backends.mps.is_available():
        return 'mps'
    else:
        return 'cpu'

# Color coding for t-SNE
class Plot_Colors(str, Enum):
    CLEAN = 'xkcd:goldenrod'
    GLAZE = 'xkcd:lavender'
    SHADE = 'xkcd:green'
    NS_GL = 'xkcd:azure'


def is_shaded_glazed(file_name: str) -> str:
    return 'glazed' in file_name and 'shaded' in file_name

def is_glazed(file_name: str) -> str:
    return 'glazed' in file_name

def is_shaded(file_name: str) -> str:
    return 'shaded' in file_name

MPATCHES = [
    mpatches.Patch(color=Plot_Colors.CLEAN, label='Clean'),
    mpatches.Patch(color=Plot_Colors.GLAZE, label='Glazed'),
    mpatches.Patch(color=Plot_Colors.SHADE, label='Shaded'),
    mpatches.Patch(color=Plot_Colors.NS_GL, label='Shaded then Glazed')
]

PERPLEXITY = 20