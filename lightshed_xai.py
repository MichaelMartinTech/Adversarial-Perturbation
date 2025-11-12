import torch
from torchvision import transforms
from PIL import Image
import os
from enum import Enum
import argparse
from lightshed_model import setup_generator, load_checkpoint
import matplotlib.pyplot as plt
import mplcursors
import matplotlib.patches as mpatches
from sklearn.manifold import TSNE
import numpy as np
import xai_utils as xu


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
        print(img_path + ' is not a valid image file')
        return None
        
def show_feature_maps(tensors: dict, n=10):
    fig, axes = plt.subplots(len(tensors), n, figsize=(15, 9))
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
    parser.add_argument('--image', help='The path to the input image')
    parser.add_argument('--folder', help='The path to the directory containing input images')
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
        if not arg_list.image:
            print('activation mode requires --image argument')
            quit()
        # Load image
        img = load_image(arg_list.image)

        # Visualize Activations
        activations = {}

        def get_activation(layer):
            def hook(model, input, output):
                activations[layer] = output.detach()
            return hook
        
        # Register hook
        generator.encoder1[2].register_forward_hook(get_activation('enc1'))
        generator.encoder2[2].register_forward_hook(get_activation('enc2'))
        generator.encoder3[2].register_forward_hook(get_activation('enc3'))
        generator.encoder4[2].register_forward_hook(get_activation('enc4'))
        generator.bottleneck[2].register_forward_hook(get_activation('btnk'))
        
        # Forward pass
        with torch.no_grad():
            img = img.to(device)
            poison = generator(img)
        
        img = img.cpu()
        poison = poison.cpu()

        show_feature_maps(activations)

    elif arg_list.mode == 'tsne':
        if not arg_list.folder:
            print('activation mode requires --folder argument')
            quit()
        
        # File handling
        directory = arg_list.folder
        if not os.path.exists(directory):
            print(f'{directory} does not exist')
        elif not os.path.isdir(directory):
            print(f'{directory} is not a directory')
        else:
            # Prepare data
            tensors = []
            colors = []
            file_names = []
            for p in os.listdir(directory):
                img = load_image(f'{directory}/{p}')
                if img is not None:
                    file_names.append(p)
                    # Build color key
                    if xu.is_shaded_glazed(p):
                        colors.append(xu.Plot_Colors.NS_GL)
                    elif xu.is_glazed(p):
                        colors.append(xu.Plot_Colors.GLAZE)
                    elif xu.is_shaded(p):
                        colors.append(xu.Plot_Colors.SHADE)
                    else:
                        colors.append(xu.Plot_Colors.CLEAN)
                    
                    # Process data / partial forward pass
                    with torch.no_grad():
                        img = img.to(device)
                        img = generator.encoder1(img)
                        img = generator.encoder2(img)
                        img = generator.encoder3(img)
                        img = generator.encoder4(img)
                        img = generator.bottleneck[0](img)
                    img = img.cpu()
                    tensors.append(img.view(-1).numpy())
            tensors_np = np.stack(tensors)

            # Visualize
            tsne = TSNE(n_components=2, perplexity=xu.PERPLEXITY, random_state=0)
            img_tsne = tsne.fit_transform(tensors_np)

            plt.figure(figsize=(8, 6))
            plot = plt.scatter(img_tsne[:, 0], img_tsne[:, 1], s=60, c=colors)

            # Show file name on hover
            cursor = mplcursors.cursor(plot, hover=True)

            @cursor.connect('add')
            def on_hover(sel):
                i = sel.index
                sel.annotation.set_text(file_names[i])
                sel.annotation.get_bbox_patch().set(fc='white', alpha=0.8)

            plt.title(f't-SNE Perplexity: {xu.PERPLEXITY}')
            plt.legend(title='Legend', handles=xu.MPATCHES, loc='best')
            plt.show()

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

    