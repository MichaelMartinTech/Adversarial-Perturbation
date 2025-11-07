import torch
from torchvision import transforms
from PIL import Image
import os
import argparse
from lightshed_model import setup_generator, load_checkpoint
import matplotlib.pyplot as plt
import numpy as np


def get_device():
    if torch.cuda.is_available():
        return 'cuda'
    elif torch.backends.mps.is_available():
        return 'mps'
    else:
        return 'cpu'

def load_image(img_path):
    extensions = {'.jpg', '.jpeg', '.png', '.JPG', '.JPEG', '.PNG'}
    if os.path.splitext(img_path)[1] in extensions:
        try:
            img = Image.open(img_path).convert('RGB')
            xform = transforms.Compose([
                transforms.Resize((512, 512)),
                transforms.ToTensor()
            ])
            img = xform(img)
            img = img.unsqueeze(0)
            return img
        except Exception as e:
            print(f'Error loading {img_path}')
            print(e)
            return None
    else:
        print('--image must be .jpg, .jpeg, or .png')
        return None
        
def show_feature_maps(tensors: dict, n=10):
    fig, axes = plt.subplots(len(tensors), n, figsize=(15, 7))
    for i, key in enumerate(tensors):
        for j in range(n):
            axes[i, j].imshow(tensors[key][0, j].cpu(), cmap='viridis')
            axes[i, j].axis('off')
            axes[i, j].set_title(f'L{i+1}, Ch{j+1}', fontsize=10)
        plt.suptitle(f'Activations per Layer')
    plt.tight_layout()
    plt.show()

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--pth', required=True, help='The path to the checkpoint file')
    parser.add_argument('--image', required=True, help='The path to the input image')
    parser.add_argument('--mode', default='activation', 
                        help="The XAI method to use. Valid arguments: 'activation', 'tsne', 'filter'")
    arg_list = parser.parse_args()

    # Check for GPU
    device = get_device()
    print(f'Device: {device}')

    # Load model
    # Model implementation hidden per LightShed authors' request
    generator, _ = setup_generator()
    checkpoint_path = arg_list.pth
    if os.path.exists(checkpoint_path):
        generator, _, _, _ = load_checkpoint(checkpoint_path, generator, None, device)
        print(f"Loaded checkpoint: {checkpoint_path}")
    else:
        raise FileNotFoundError(f"No checkpoint found at {checkpoint_path}")   
    generator.eval()

    if arg_list.mode == 'activation':
        # Load image
        img = load_image(arg_list.image)

        # Visualize Activations
        activations = {}

        def get_activation(layer):
            def hook(model, input, output):
                activations[layer] = output.detach()
            return hook
        
        # Register hook
        generator.encoder1[0].register_forward_hook(get_activation('enc1'))
        generator.encoder2[0].register_forward_hook(get_activation('enc2'))
        generator.encoder3[0].register_forward_hook(get_activation('enc3'))
        generator.encoder4[0].register_forward_hook(get_activation('enc4'))
        
        # Forward pass
        with torch.no_grad():
            img = img.to(device)
            poison = generator(img)
        
        img = img.cpu()
        poison = poison.cpu()

        show_feature_maps(activations)

    elif arg_list.mode == 'tsne':
        print('Work in progress')

    elif arg_list.mode == 'filter':
        enc1 = generator.encoder1[0]
        filters = enc1.weight.data.clone().cpu()

        # Normalize
        filters_min = filters.min()
        filters_max = filters.max()
        filters = (filters - filters_min) / (filters_max - filters_min)

        # Visualize
        num_filters = filters.shape[0]
        fig, axes = plt.subplots(8, num_filters // 8, figsize=(5, 5))

        for i, ax in enumerate(axes.flat):
            f = filters[i].permute(1, 2, 0).numpy()
            f = np.clip(f, 0, 1)
            ax.imshow(f)
            ax.axis('off')

        plt.tight_layout(rect=[0, 0, 1, 0.95])
        plt.suptitle("Encoder Layer Filters")
        plt.show()

    