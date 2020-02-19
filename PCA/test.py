import numpy as np
import cv2
import matplotlib.pyplot as plt
from sklearn.decomposition import PCA, IncrementalPCA
import pickle
from sys import getsizeof
import copy

from os import listdir
from os.path import isfile, join
from pca import *
from skimage.metrics import structural_similarity as ssim
import matplotlib.pyplot as plt

test_images_dir = "../data/PCA"
img_paths = [join(test_images_dir, f) for f in listdir(test_images_dir) if isfile(join(test_images_dir, f))]

test_variances = [95, 99, 99.99]

results = []

for img_path in img_paths:
    raw_img = cv2.imread(img_path)
    num_channels = raw_img.shape[2]

    output = [cv2.cvtColor(raw_img, cv2.COLOR_BGR2RGB)]
    separate_channels = []

    # separating RGB channels
    if raw_img.ndim == 3:
        for i in range(num_channels):
            separate_channels.append(raw_img[:,:,i])
    else:
        separate_channels.append(raw_img)

    std = list(map(np.std, separate_channels))
    mean = list(map(np.mean, separate_channels))

    # preprocess image force image 
    separate_channels = list(map(lambda x,m,s: (x-m)/s, separate_channels, mean, std))
    
    img = ((raw_img / 255.000) - 0.500).astype(np.float16)

    for var in test_variances:

        # getting required number of components for each channel
        num_components = list(map(required_num_components, separate_channels, 
                            np.ones(len(separate_channels), dtype=np.float) * var))
        
        print("Num Components: {}".format(num_components))

        # compressing each channel

        compressed_data = list(map(pca_compress_channel, separate_channels, num_components))

        # computing the size and compression ratio
        
        sz_img = img.nbytes
        sz_compressed_data = 0
        for (X_reduced, ipca) in compressed_data:
            sz_compressed_data += (X_reduced.nbytes + getsizeof(ipca))

        compression_ratio = sz_img/sz_compressed_data

        print("Size of the compressed data: {}".format(sz_compressed_data))
        print("Size of the raw data: {}".format(sz_img))
        print("Compression Ratio: {}".format(compression_ratio))

        # decompressed image
        decompressed_data = []
        for (X_reduced, ipca) in compressed_data:
            decompressed_data.append(pca_decompress_channel(X_reduced, ipca))

        im_shape = decompressed_data[-1].shape

        # reconstructing image
        if num_channels == 1:
            recon_im = np.zeros(shape=im_shape, dtype=np.uint8)
            recon_im = decompressed_data * std[-1] + mean[-1]
        else:
            recon_im = np.zeros(shape=im_shape + tuple([num_channels]), dtype=np.float16)
            
            for i in range(num_channels):
                recon_im[:,:,i] = decompressed_data[i] * std[i] + mean[i]
        
        # recon_im = (recon_im + 0.500) * 255.0
        recon_im = recon_im.astype(np.uint8)

        # saving images for comparison
        # output_path = args.output + "-recon-{}.jpg".format(args.var)
        # cv2.imwrite(output_path, recon_im)

        ssim_curr = ssim(raw_img, recon_im, data_range=recon_im.max() - recon_im.min(), multichannel=True)
        print("SSIM : {}".format(ssim_curr))

        output.append((cv2.cvtColor(recon_im, cv2.COLOR_BGR2RGB), compression_ratio, ssim_curr))
    
    results.append(output)



fig, ax = plt.subplots( len(results),  len(test_variances) + 1, figsize=(25,10))

for i in range(len(results)):
    for j in range(len(test_variances) + 1):
        data = results[i][j]
        ax[i, j].axis('off')

        if j != 0:  # reconstructed image
            im, cr, ssm = data
            ax[i, j].imshow(im)
            ax[i, j].set_title("CR: {}, SSIM:{}".format(np.round(cr, 3), np.round(ssm, 2)))
        else:       # ground truth
            im = data 
            ax[i,j].imshow(im)
            ax[i,j].set_title("Ground Truth")

plt.tight_layout()
plt.show()
