import argparse
import numpy as np 
from matplotlib.image import imread
import matplotlib.pyplot as plt
from sklearn.decomposition import PCA, IncrementalPCA

def required_components(variance=90.0):
    return 0


def main():
    parser = argparse.ArgumentParser(description='Using the PCA for image compression.')
    parser.add_argument('input', type=str, help='Path for input image')
    parser.add_argument('output', type=str, help='Path for output image')
    parser.add_argument('--var', '--total_variance', type=float, default=90.0, 
                        help='Total variance % to be covered by PCA')

    args = parser.parse_args()

    print("Input Image: {}".format(args.input))
    print("Output Image: {}".format(args.output))
    print("Total Variance: {}".format(args.var))

    k = required_components(args.var)


if __name__=="__main__":
    main()