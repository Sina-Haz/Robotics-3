import argparse
from create_scene import load_polygons, save_polygons

# Usage: python3 crawler.py --file estim1/estim1_X_Y_Z_N.npy
if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='This file is meant to crawl through all estimates and clip them by 1 b/c we accidentally saved an extra pose')
    parser.add_argument('--file', required=True,help='Filename you want us to crawl thru')
    args = parser.parse_args()

    vals = load_polygons(args.file)
    vals = vals[0:201, :]
    save_polygons(vals, args.file)

