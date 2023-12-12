import argparse
from create_scene import load_polygons, save_polygons

# Usage: python3 crawler.py --file estim1/estim1_X_Y_Z_N.npy --gt gts/gt_X_Y.npy
if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='This file is meant to crawl through all estimates and clip them by 1 b/c we accidentally saved an extra pose')
    parser.add_argument('--file', required=True,help='Filename you want us to crawl thru')
    parser.add_argument('--gt', required=False, help='to ensure estim1 starts at correct initial pose')
    args = parser.parse_args()

    gts = load_polygons(args.gt)
    vals = load_polygons(args.file)
    vals[0] = gts[0]
    save_polygons(vals, args.file)

