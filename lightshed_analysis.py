import os
import argparse
import pandas as pd
import matplotlib.pyplot as plt
from typing import Union
from xai_utils import BOX_LEGEND_HANDLES, GENERAL_COLOR_LIST

noise_entropy = {}
noise_detect_rate = {}
mask_entropy = {}
mask_detect_rate = {}
lightness_entropy = {}
lightness_detect_rate = {}

color_mapper = {}
color_index = 0
color_list = []
entropies = []
file_sizes = []

def store_info(struct: dict, key: str, value: Union[int, float]) -> None:
    if key not in struct:
        struct[key] = []
    struct[key].append(value)

def print_entropy(struct: dict) -> None:
    for key in sorted(struct.keys()):
        print(f'Average entropy of {key}: {sum(struct[key]) / len(struct[key])}')

def print_detect(struct: dict) -> None:
    for key in sorted(struct.keys()):
        print(f'Detection rate of {key}: {sum(struct[key]) / len(struct[key]) * 100}%')

def plot_NML() -> None:
    fig, ax = plt.subplots(1, 3, figsize=(15, 5))

    plt.suptitle('Shannon Entropy for Reconstructed Poison')

    ax[0].boxplot([noise_entropy[key] for key in sorted(noise_entropy.keys())],
                    tick_labels=[key for key in sorted(noise_entropy.keys())],
                    orientation='horizontal', showmeans=True, meanline=True)
    ax[0].set_xlabel('Entropy')
    ax[0].set_ylabel('Noise')

    ax[1].boxplot([mask_entropy[key] for key in sorted(mask_entropy.keys())],
                    tick_labels=[key for key in sorted(mask_entropy.keys())],
                    orientation='horizontal', showmeans=True, meanline=True)
    ax[1].set_xlabel('Entropy')
    ax[1].set_ylabel('Mask')

    ax[2].boxplot([lightness_entropy[key] for key in sorted(lightness_entropy.keys())],
                    tick_labels=[key for key in sorted(lightness_entropy.keys())],
                    orientation='horizontal', showmeans=True, meanline=True)
    ax[2].set_xlabel('Entropy')
    ax[2].set_ylabel('Lightness')

    fig.legend(handles=BOX_LEGEND_HANDLES, loc='center left')

    plt.tight_layout()
    fig.subplots_adjust(left=0.18, wspace=0.4)
    plt.show()

def plot_compressibility() -> None:
    plt.scatter(file_sizes, entropies, c=color_list)
    plt.tight_layout()
    plt.show()


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--csv', required=True, help='The path to the csv output of LightShed to analyze')
    parser.add_argument('--masks', default=os.path.join(os.getcwd(), 'noise_data', 'masks'), help='Directory containing masks')
    arg_list = parser.parse_args()

    if not os.path.exists(arg_list.csv) and not os.path.splitext(arg_list.csv)[1] == '.csv':
        raise FileNotFoundError(f'{arg_list.csv} not found or is not a CSV')
    if not os.path.exists(arg_list.masks) and not os.path.isdir(arg_list.masks):
        raise FileNotFoundError(f'{arg_list.masks} not found or is not directory')

    if os.path.exists(arg_list.csv) and os.path.splitext(arg_list.csv)[1] == '.csv':
        info = pd.read_csv(arg_list.csv)
        for row in info.itertuples(index=False):
            filename = os.path.splitext(row[0])[0]
            entropy = row[1]
            detected = 1 if row[2] == 'tensor(True)' else 0
            
            method = filename.split('_')
            noise = method[1]
            mask = method[2]
            lightness = method[3]

            store_info(noise_entropy, noise, entropy)
            store_info(noise_detect_rate, noise, detected)
            store_info(mask_entropy, mask, entropy)
            store_info(mask_detect_rate, mask, detected)
            store_info(lightness_entropy, lightness, entropy)
            store_info(lightness_detect_rate, lightness, detected)

            if noise not in color_mapper:
                color_mapper[noise] = GENERAL_COLOR_LIST[color_index]
                color_index += 1
            color_list.append(color_mapper[noise])

            entropies.append(entropy)
            mask_file = os.path.join(arg_list.masks, f'{mask}_{lightness}.png')
            file_sizes.append(os.path.getsize(mask_file))
        
        print_entropy(noise_entropy)
        print_detect(noise_detect_rate)
        print_entropy(mask_entropy)
        print_detect(mask_detect_rate)
        print_entropy(lightness_entropy)
        print_detect(lightness_detect_rate)

        plot_NML()
        # print(color_mapper)
        # plot_compressibility()
        

            