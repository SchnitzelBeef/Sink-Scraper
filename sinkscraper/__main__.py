import argparse
import os
import sys
from .sinkscraper import buildScraper
sys.path.append("./FBA_Matting")

print("\n=== SinkScraper started ===\n")

parser = argparse.ArgumentParser(
    prog='SinkScraper',
    description='Creates a 3D printable scraper model from a picture of a person.',
    epilog='Made for fun!')

parser.add_argument('filename', help='file name of the image to process')
parser.add_argument('-sh','--show', help='show intermediate images', required=False, action='store_true')
parser.add_argument('-sa','--save', help='save intermediate pictures created', required=False, action='store_true')
parser.add_argument('-nc','--no-cut', help='do not cut out the person from the image', required=False, action='store_true')
args = parser.parse_args()

# check if the file exists
os.chdir("./pictures/")
try:
    with open(args.filename, 'r') as f:
        pass
except FileNotFoundError:
    print(f"Error: File '{args.filename}' not found.")
    sys.exit(1)
os.chdir("../")

model_builder = buildScraper(args.filename, save_intermediate_pictures=args.save, show_intermediate_pictures=args.show, no_cut=args.no_cut, log=True)
model_image = model_builder.applyModel()
model_builder.finishModel(model_image) 

sys.exit(0)
    