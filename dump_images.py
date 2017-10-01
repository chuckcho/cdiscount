# This Python 3 environment comes with many helpful analytics libraries installed
# It is defined by the kaggle/python docker image: https://github.com/kaggle/docker-python
# For example, here's several helpful packages to load in

#import numpy as np # linear algebra
import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)

# Input data files are available in the "../input/" directory.
# For example, running this (by clicking run or pressing Shift+Enter) will list the files in the input directory

data_dir = '/media/6TB/cdiscount'

# Any results you write to the current directory are saved as output.
import io
import bson                       # this is installed with the pymongo package
#import matplotlib.pyplot as plt
#from skimage.data import imread   # or, whatever image library you prefer
#from skimage.io import imsave
#import multiprocessing as mp      # will come in handy due to the size of the data
import os
from PIL import Image

# Simple data processing
#data_prefix = 'train_example'
data_prefix = 'train'
bson_file = os.path.join(data_dir, data_prefix+'.bson')
out_dir = '/media/6TB/cdiscount/images/{}'.format(data_prefix)
prod_to_category_file = '/media/6TB/cdiscount/{}_prod_id_to_category.csv'.format(data_prefix)
verbose = True

data = bson.decode_file_iter(open(bson_file, 'rb'))
n_samples = sum([1 for x in data])
print "#examples={}".format(n_samples)
data = bson.decode_file_iter(open(bson_file, 'rb'))

if not os.path.exists(out_dir):
    os.makedirs(out_dir)

if 'train' in data_prefix:
    prod_to_category = dict()

for count, d in enumerate(data):
    product_id = d['_id']
    if 'train' in data_prefix:
        category_id = d['category_id']
        prod_to_category[product_id] = category_id
    for e, pic in enumerate(d['imgs']):
        #picture = imread(io.BytesIO(pic['picture']))
        picture = Image.open(io.BytesIO(pic['picture']))

        out_file = os.path.join(out_dir, '{:08d}_{:08d}_{}.png'.format(count, product_id, category_id))
        #imsave(out_file, picture)
        picture.save(out_file)
    if verbose and count % 10000 == 0:
        print "[Info] processed {}% (count={} out of {})".format(int(float(count)/n_samples*100), count, n_samples)

if 'train' in data_prefix:
    prod_to_category = pd.DataFrame.from_dict(prod_to_category, orient='index')
    prod_to_category.index.name = '_id'
    prod_to_category.rename(columns={0: 'category_id'}, inplace=True)

    print prod_to_category.head()
    prod_to_category.to_csv(prod_to_category_file)

# run:
#   ls *png | egrep -v "_trimmed" | sed -e 's/\.png//' | xargs -I {} convert {}.png -fuzz 1% -trim +repage {}_trimmed.png
