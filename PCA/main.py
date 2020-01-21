import argparse
import numpy as np 
from matplotlib.image import imread
import matplotlib.pyplot as plt

def main():
    parser = argparse.ArgumentParser(description='Using the PCA for image compression.')
    parser.add_argument('input', type=str, help='Path for input image')
    parser.add_argument('output', type=str, help='Path for output image')

    args = parser.parse_args()

    print("Input Image: {}".format(args.input))
    print("Output Image: {}".format(args.output))

    

if __name__=="__main__":
    main()