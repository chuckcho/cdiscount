Steps to prep / train
=====================
1. Dump images from BSON
2. Trim white BG from images
3. Convert images to sharded tfrecords
4. Download pre-trained model: tensorflow-resnet-pretrained-20160509.tar.gz / http://download.tensorflow.org/models/inception_resnet_v2_2016_08_30.tar.gz

Trimmed white background by:
```
mkdir ../train-trimmed
ls | egrep "\.png" | xargs -I {} convert {} -fuzz 1% -trim +repage ../train-trimmed/{}
```
https://github.com/tensorflow/models/tree/master/research/inception
https://github.com/tensorflow/models/blob/master/research/inceptioun/inception/data/build_imagenet_data.py
