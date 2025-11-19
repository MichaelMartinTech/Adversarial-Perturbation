import os
import argparse
import pandas as pd

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--csv', required=True, help='The path to the csv output of LightShed to analyze')
    arg_list = parser.parse_args()

    noise_entropy = {}
    noise_detect_rate = {}
    mask_entropy = {}
    mask_detect_rate = {}
    lightness_entropy = {}
    lightness_detect_rate = {}

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

            if noise not in noise_entropy:
                noise_entropy[noise] = []
            noise_entropy[noise].append(entropy)

            if noise not in noise_detect_rate:
                noise_detect_rate[noise] = []
            noise_detect_rate[noise].append(detected)

            if mask not in mask_entropy:
                mask_entropy[mask] = []
            mask_entropy[mask].append(entropy)

            if mask not in mask_detect_rate:
                mask_detect_rate[mask] = []
            mask_detect_rate[mask].append(detected)

            if lightness not in lightness_entropy:
                lightness_entropy[lightness] = []
            lightness_entropy[lightness].append(entropy)

            if lightness not in lightness_detect_rate:
                lightness_detect_rate[lightness] = []
            lightness_detect_rate[lightness].append(detected)
        
        for key, value in noise_entropy.items():
            print(f'Average entropy of {key}: {sum(value) / len(value)}')
        
        for key, value in noise_detect_rate.items():
            print(f'Detection rate of {key}: {sum(value) / len(value) * 100}%')
        
        for key, value in mask_entropy.items():
            print(f'Average entropy of {key}: {sum(value) / len(value)}')
        
        for key, value in mask_detect_rate.items():
            print(f'Detection rate of {key}: {sum(value) / len(value) * 100}%')
        
        for key, value in lightness_entropy.items():
            print(f'Average entropy of {key}: {sum(value) / len(value)}')
        
        for key, value in lightness_detect_rate.items():
            print(f'Detection rate of {key}: {sum(value) / len(value) * 100}%')
        

            