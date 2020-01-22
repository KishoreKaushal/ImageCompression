import argparse
import numpy as np
import cv2
import matplotlib.pyplot as plt
from sklearn.decomposition import PCA, IncrementalPCA
import pickle
from sys import getsizeof
import copy

DEFAULT_VARIANCE_PERCENT = 95.0

def required_num_components(single_channel_img, variance_percentage=DEFAULT_VARIANCE_PERCENT):
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


def pca_decompress_channel(X_reduced, incr_pca):
    return incr_pca.inverse_transform(X_reduced)


def main():
    parser = argparse.ArgumentParser(description='Using the PCA for image compression.')
    parser.add_argument('input', type=str, help='Path for input image')
    parser.add_argument('output', type=str, help='Path for output data')
    parser.add_argument('--var', '--total_variance', type=float, default=DEFAULT_VARIANCE_PERCENT, 
                        help='Total variance % to be covered by PCA')

    args = parser.parse_args()

    print("Input Image: {}".format(args.input))
    print("Output Image: {}".format(args.output))
    print("Total Variance: {}".format(args.var))

    raw_img = cv2.imread(args.input)
    num_channels = raw_img.shape[2]

    # preprocess image force image 
    img = ((raw_img / 255.000) - 0.500).astype(np.float16)

    separate_channels = []

    # separating RGB channels
    if img.ndim == 3:
        for i in range(num_channels):
            separate_channels.append(img[:,:,i])
    else:
        separate_channels.append(img)

    # getting required number of components for each channel
    num_components = list(map(required_num_components, separate_channels, 
                        np.ones(len(separate_channels), dtype=np.float) * args.var))
    
    print("Num Components: {}".format(num_components))

    # compressing each channel

    # sz_compressed_data = 0
    # compressed_data = []
    # for (X, k) in zip(separate_channels, num_components):
    #     compressed_data.append(pca_compress_channel(X,k))
    #     print(compressed_data[-1][0].shape)
    #     print("sz : {}".format(getsizeof(compressed_data[-1][0])))

    compressed_data = list(map(pca_compress_channel, separate_channels, num_components))

    # writing the compressed data to the file
    # with open(args.output, 'wb') as fout:
    #     pickle.dump(compressed_data, fout)

    # computing the size and compression ratio
    
    sz_img = img.nbytes
    sz_compressed_data = 0
    for (X_reduced, ipca) in compressed_data:
        sz_compressed_data += (X_reduced.nbytes + getsizeof(ipca))

    print("Size of the compressed data: {}".format(sz_compressed_data))
    print("Size of the raw data: {}".format(sz_img))
    print("Compression Ratio: {}".format(sz_img/sz_compressed_data))

    # decompressed image
    decompressed_data = []
    for (X_reduced, ipca) in compressed_data:
        decompressed_data.append(pca_decompress_channel(X_reduced, ipca))

    im_shape = decompressed_data[-1].shape

    # reconstructing image
    if num_channels == 1:
        recon_im = np.zeros(shape=im_shape, dtype=np.uint8)
        decompressed_data
    else:
        recon_im = np.zeros(shape=im_shape + tuple([num_channels]), dtype=np.float16)
        
        for i in range(num_channels):
            recon_im[:,:,i] = decompressed_data[i]
    
    recon_im = (recon_im + 0.500) * 255.0
    recon_im = recon_im.astype(np.uint8)

    # saving images for comparison
    output_path = args.output + "-recon-{}.jpg".format(args.var)
    cv2.imwrite(output_path, recon_im)


if __name__=="__main__":
    main()