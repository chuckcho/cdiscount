Steps to prep / train
=====================
1. Dump images from BSON
2. Trim white BG from images: `ls | egrep "\.png" | xargs -I {} convert {} -fuzz 1% -trim +repage ../train-trimmed/{}`
3. Convert images to sharded tfrecords
4. Download pre-trained model: `wget http://download.tensorflow.org/models/inception_resnet_v2_2016_08_30.tar.gz`
