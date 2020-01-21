import argparse
import numpy as np
import cv2
import matplotlib.pyplot as plt
from sklearn.decomposition import PCA, IncrementalPCA
import pickle
from sys import getsizeof

def required_num_components(single_channel_img, variance_percentage=90.0):
    img = np.array(single_channel_img, dtype=np.float)
    assert img.ndim == 2, ("2D tensors are expected")
    
    pca = PCA()
    pca.fit(img)
    cumulative_variance = np.cumsum(pca.explained_variance_ratio_) * 100

    # How many PC's can represent {variance} % of image?
    num_components = np.argmax(cumulative_variance > variance_percentage)
    return num_components
    

def pca_compress_channel(X, k):
    incr_pca = IncrementalPCA(n_components=k)
    X_reduced = incr_pca.fit_transform(X).astype(np.float16)
    return (X_reduced, incr_pca)


def pca_decompress_channel():
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

    raw_img = cv2.imread(args.input)

    # preprocess image force image 
    img = ((raw_img / 255.000) - 0.500).astype(np.float16)

    separate_channels = []

    # separating RGB channels
    if img.ndim == 3:
        for i in range(img.shape[2]):
            separate_channels.append(img[:,:,i])
    else:
        separate_channels.append(img)

    # getting required number of components for each channel
    num_components = list(map(required_num_components, separate_channels, 
                        np.ones(len(separate_channels), dtype=np.float) * args.var))

    # compressing each channel
    compressed_data = list(map(pca_compress_channel, separate_channels, num_components))

    # writing the compressed data to the file
    with open(args.output, 'wb') as fout:
        pickle.dump(compressed_data, fout)

    sz_compressed_data = getsizeof(compressed_data)
    sz_img = img.nbytes

    print("Size of the compressed data: {}".format(sz_compressed_data))
    print("Size of the raw data: {}".format(sz_img))
    print("Compression Ratio: {}".format(sz_img/sz_compressed_data))


if __name__=="__main__":
    main()