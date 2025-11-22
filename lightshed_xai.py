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

extensions = {'.jpg', '.jpeg', '.png', '.JPG', '.JPEG', '.PNG'}

def load_image(img_path: str, unsqueeze: bool = True) -> Image:    
    if os.path.splitext(img_path)[1] in extensions:
        try:
            img = Image.open(img_path).convert('RGB')
            xform = transforms.Compose([
                transforms.Resize((512, 512)),
                transforms.ToTensor()
            ])
            img = xform(img)
            if unsqueeze:
                img = img.unsqueeze(0)
            return img
        except Exception as e:
            print(f'Error loading {img_path}')
            print(e)
            return None
    else:
        print(img_path + ' is not a valid image file')
        return None
    
def load_multi_images(img_paths: list[str]) -> tuple[torch.Tensor, list[str]]:
    images = []
    file_names = []
    for imgpath in img_paths:
        img = load_image(imgpath, unsqueeze=False)
        if img is not None:
            images.append(img)
            file_names.append(os.path.basename(imgpath))
    return torch.stack(images), file_names


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--pth', required=True, help='The path to the checkpoint file')
    parser.add_argument('--images', nargs='+', help='The path(s) to the input image(s)')
    parser.add_argument('--folder', help='The path to the directory containing input images')
    parser.add_argument('--mode', default='activation', 
                        help="The XAI method to use. Valid arguments: 'activation', 'tsne', 'filter'")
    arg_list = parser.parse_args()

    # Check for GPU
    device = xu.get_device()
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
        if not arg_list.images:
            print('activation mode requires --images argument')
            quit()
        
        # Load image
        images, file_names = load_multi_images(arg_list.images)

        if len(file_names) < 1:
            print('No valid images provided')
            quit()

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
            images = images.to(device)
            poison = generator(images)
        
        images = images.cpu()
        poison = poison.cpu()
        num_layers = len(activations)

        fig = plt.figure(figsize=(15,9))
        current_index = 0

        def show_feature_maps(n=10):
            fig.clear()
            for i, (_, tensor) in enumerate(activations.items()):
                for j in range(n):
                    ax = fig.add_subplot(num_layers, n, i * n + j + 1)
                    fmap = tensor[current_index, j].cpu()
                    ax.imshow(fmap, cmap='viridis')
                    ax.axis('off')
                    ax.set_title(f'L{i+1}, Ch{j+1}', fontsize=10)
            fig.suptitle(f'Activations per Layer for {file_names[current_index]}\n{swipe_instruct}')
            fig.canvas.draw_idle()
            plt.tight_layout()

        def on_key(event):
            global current_index
            if event.key == 'right':
                current_index = (current_index + 1) % num_images
                show_feature_maps()
            elif event.key == 'left':
                current_index = (current_index - 1) % num_images
                show_feature_maps()

        num_images = next(iter(activations.values())).shape[0]
        swipe_instruct = '(Press Left or Right keys to switch images)' if num_images > 1 else ''
       
        fig.canvas.mpl_connect('key_press_event', on_key)
        show_feature_maps()

        plt.show()

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

    