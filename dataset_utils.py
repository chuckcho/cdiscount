import math
import os
import sys
import tensorflow as tf
import glob

# slim = tf.contrib.slim
#===================================================  Dataset Utils  ===================================================

def int64_feature(values):
  """Returns a TF-Feature of int64s.

  Args:
    values: A scalar or list of values.

  Returns:
    a TF-Feature.
  """
  if not isinstance(values, (tuple, list)):
    values = [values]
  return tf.train.Feature(int64_list=tf.train.Int64List(value=values))


def bytes_feature(values):
  """Returns a TF-Feature of bytes.

  Args:
    values: A string.

  Returns:
    a TF-Feature.
  """
  return tf.train.Feature(bytes_list=tf.train.BytesList(value=[values]))


def image_to_tfexample(image_data, image_format, height, width, class_serial):
  return tf.train.Example(features=tf.train.Features(feature={
      'image/encoded': bytes_feature(image_data),
      'image/format': bytes_feature(image_format),
      'image/class': int64_feature(class_serial),
      'image/height': int64_feature(height),
      'image/width': int64_feature(width),
  }))

def write_label_file(serial_to_class_ids, dataset_dir,
                     filename='labels.csv'):
  """Writes a file with the list of class ids.

  Args:
    seral_to_class_ids: A map of (integer) serials to class ids.
    dataset_dir: The directory in which the labels file should be written.
    filename: The filename where the class names are written.
  """
  labels_filename = os.path.join(dataset_dir, filename)
  with tf.gfile.Open(labels_filename, 'w') as f:
    for serial in serial_to_class_ids:
      class_id = serial_to_class_ids[serial]
      f.write('%d, %d\n' % (serial, class_id))

#=======================================  Conversion Utils  ===================================================

# Create an image reader object for easy reading of the images
class ImageReader(object):
  """Helper class that provides TensorFlow image coding utilities."""

  def __init__(self):
    # Initializes function that decodes RGB PNG data.
    self._decode_png_data = tf.placeholder(dtype=tf.string)
    self._decode_png = tf.image.decode_png(self._decode_png_data, channels=3)

  def read_image_dims(self, sess, image_data):
    image = self.decode_png(sess, image_data)
    return image.shape[0], image.shape[1]

  def decode_png(self, sess, image_data):
    image = sess.run(self._decode_png,
                     feed_dict={self._decode_png_data: image_data})
    assert len(image.shape) == 3
    assert image.shape[2] == 3
    return image

def _get_filenames_and_classes(dataset_dir, img_ext='png'):
  """Returns a list of filenames and inferred class names.

  Args:
    dataset_dir: A directory containing all images. Image filename itself
    encodes class names.

  Returns:
    A list of image file paths, relative to `dataset_dir` and the list of
    class names.
  """
  photo_filenames = glob.glob(os.path.join(dataset_dir, '*.' + img_ext))
  class_ids = set()
  for f in photo_filenames:
    fname, _ = os.path.splitext(os.path.basename(f))
    class_id = int(fname.split('_')[2])
    class_ids.add(class_id)
    #print "[Debug] file={}, class_id={}".format(f, class_id)

  return photo_filenames, sorted(list(class_ids))


def _get_dataset_filename(dataset_dir, split_name, shard_id, tfrecord_filename, _NUM_SHARDS):
  output_filename = '%s_%s_%04d_of_%04d.tfrecord' % (
      tfrecord_filename, split_name, shard_id, _NUM_SHARDS)
  return os.path.join(dataset_dir, output_filename)


def _convert_dataset(split_name, filenames, class_ids_to_serial, dataset_dir,
                tfrecord_filename, _NUM_SHARDS, simulate=False):
  """Converts the given filenames to a TFRecord dataset.

  Args:
    split_name: The name of the dataset, either 'train' or 'validation'.
    filenames: A list of absolute paths to png images.
    class_ids_to_serial: A dictionary from class ids (int) to serial (int).
    dataset_dir: The directory where the converted datasets are stored.
    simulate: Just do simulation
  """
  assert split_name in ['train', 'validation']

  num_per_shard = int(math.ceil(len(filenames) / float(_NUM_SHARDS)))

  with tf.Graph().as_default():
    image_reader = ImageReader()

    with tf.Session('') as sess:

      for shard_id in range(_NUM_SHARDS):
        output_filename = _get_dataset_filename(
            dataset_dir, split_name, shard_id, tfrecord_filename = tfrecord_filename, _NUM_SHARDS = _NUM_SHARDS)

        if not simulate:
          tfrecord_writer = tf.python_io.TFRecordWriter(output_filename)

        start_ndx = shard_id * num_per_shard
        end_ndx = min((shard_id+1) * num_per_shard, len(filenames))
        for i in range(start_ndx, end_ndx):
          sys.stdout.write('\r>> Converting image %d/%d shard %d' % (
              i+1, len(filenames), shard_id))
          sys.stdout.flush()

          # Read the filename:
          image_data = tf.gfile.FastGFile(filenames[i], 'r').read()
          height, width = image_reader.read_image_dims(sess, image_data)

          fname, _ = os.path.splitext(os.path.basename(filenames[i]))
          class_id = int(fname.split('_')[2])
          class_serial = class_ids_to_serial[class_id]
          example = image_to_tfexample(
              image_data, 'png', height, width, class_serial)
          if not simulate:
            tfrecord_writer.write(example.SerializeToString())

  sys.stdout.write('\n')
  sys.stdout.flush()

def _dataset_exists(dataset_dir, _NUM_SHARDS, output_filename):
  for split_name in ['train', 'validation']:
    for shard_id in range(_NUM_SHARDS):
      tfrecord_filename = _get_dataset_filename(
          dataset_dir, split_name, shard_id, output_filename, _NUM_SHARDS)
      if not tf.gfile.Exists(tfrecord_filename):
        return False
  return True
