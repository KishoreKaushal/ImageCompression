import argparse
import numpy as np
import cv2
import matplotlib.pyplot as plt
from sklearn.decomposition import PCA, IncrementalPCA
import pickle

def required_num_components(single_channel_img, variance_percentage=90.0):
    img = np.array(single_channel_img, dtype=np.float)
    assert img.ndim == 2, ("2D tensors are expected")
    
    pca = PCA()
    pca.fit(img)
    cumulative_variance = np.cumsum(pca.explained_variance_ratio_) * 100

    # How many PC's can represent {variance} % of image?
    num_components = np.argmax(cumulative_variance > variance_percentage)
    return num_components
    

def pca_compress_component(X, k):
    incr_pca = IncrementalPCA(n_components=k)
    X_reduced = incr_pca.fit_transform(X)
    return (X_reduced, incr_pca)


def pca_decompress_component():    
    pass


def main():
    parser = argparse.ArgumentParser(description='Using the PCA for image compression.')
    parser.add_argument('input', type=str, help='Path for input image')
    parser.add_argument('output', type=str, help='Path for output data')
    parser.add_argument('--var', '--total_variance', type=float, default=90.0, 
                        help='Total variance % to be covered by PCA')

    args = parser.parse_args()

    print("Input Image: {}".format(args.input))
    print("Output Image: {}".format(args.output))
    print("Total Variance: {}".format(args.var))

    img = cv2.imread(args.input)

    separate_channels = []

    # separating RGB channels
    if img.ndim == 3:
        for i in range(img.shape[2]):
            separate_channels.append(img[:,:,i])
    else:
        separate_channels.append(img)

    num_components = list(map(required_num_components, separate_channels, 
                        np.ones(len(separate_channels), dtype=np.float) * args.var))


if __name__=="__main__":
    main()