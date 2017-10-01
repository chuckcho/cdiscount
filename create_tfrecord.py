import random
import tensorflow as tf
from dataset_utils import _dataset_exists, _get_filenames_and_classes, _convert_dataset, write_label_file
import os

#====================================================DEFINE YOUR ARGUMENTS=======================================================================
flags = tf.app.flags

# State your dataset directory
flags.DEFINE_string('dataset_dir', '/media/6TB/cdiscount/images/train-trimmed', 'String: Your dataset directory')

# State your output tfrecord directory
flags.DEFINE_string('tfrecord_dir', '/media/6TB/cdiscount/tfrecord', 'String: Your output directory')

# The number of images in the validation set. You would have to know the total number of examples in advance. This is essentially your evaluation dataset.
flags.DEFINE_float('validation_size', 0.2, 'Float: The proportion of examples in the dataset to be used for validation')

# The number of shards to split the dataset into
flags.DEFINE_integer('num_shards', 1024, 'Int: Number of shards to split the TFRecord files')

# Seed for repeatability.
flags.DEFINE_integer('random_seed', 9257042, 'Int: Random seed to use for repeatability.')

# Output filename for the naming the TFRecord file
flags.DEFINE_string('tfrecord_filename', 'cdiscount', 'String: The output filename to name your TFRecord file')

# Just simulate and don't write files
flags.DEFINE_boolean('simulate', False, 'Boolean: just simulate')

FLAGS = flags.FLAGS

def main():

    #==============================================================CHECKS==========================================================================
    # Check if there is a tfrecord_filename entered
    if not FLAGS.tfrecord_filename:
        raise ValueError('tfrecord_filename is empty. Please state a tfrecord_filename argument.')

    # Check if there is a dataset directory entered
    if not FLAGS.dataset_dir:
        raise ValueError('dataset_dir is empty. Please state a dataset_dir argument.')

    # Check if there is a dataset directory entered
    if not FLAGS.tfrecord_dir:
        raise ValueError('tfrecord_dir is empty. Please state a tfrecord_dir argument.')

    if not os.path.exists(FLAGS.tfrecord_dir):
        os.makedirs(FLAGS.tfrecord_dir)

    # If the TFRecord files already exist in the directory, then exit without creating the files again
    if _dataset_exists(
            dataset_dir = FLAGS.tfrecord_dir,
            _NUM_SHARDS = FLAGS.num_shards,
            output_filename = FLAGS.tfrecord_filename):
        print 'Dataset files already exist. Exiting without re-creating them.'
        return None
    #==============================================================END OF CHECKS===================================================================

    # Get a list of photo_filenames and a list of sorted class names from parsing the subdirectories.
    photo_filenames, class_ids = _get_filenames_and_classes(FLAGS.dataset_dir)

    # Refer each of the class name to a specific integer number for predictions later
    class_ids_to_serial = dict(zip(class_ids, range(len(class_ids))))

    # Write the labels file:
    serial_to_class_ids = dict(zip(range(len(class_ids)), class_ids))
    write_label_file(serial_to_class_ids, FLAGS.dataset_dir)

    # Find the number of validation examples we need
    num_validation = int(float(FLAGS.validation_size) * len(photo_filenames))

    # Divide the training datasets into train and test:
    random.seed(FLAGS.random_seed)
    random.shuffle(photo_filenames)
    training_filenames = photo_filenames[num_validation:]
    validation_filenames = photo_filenames[:num_validation]

    # Convert the training and validation sets.
    _convert_dataset('train', training_filenames, class_ids_to_serial,
                     dataset_dir = FLAGS.tfrecord_dir, tfrecord_filename = FLAGS.tfrecord_filename, _NUM_SHARDS = FLAGS.num_shards,
                     simulate = FLAGS.simulate)
    _convert_dataset('validation', validation_filenames, class_ids_to_serial,
                     dataset_dir = FLAGS.tfrecord_dir, tfrecord_filename = FLAGS.tfrecord_filename, _NUM_SHARDS = FLAGS.num_shards,
                     simulate = FLAGS.simulate)

    print '\nFinished converting the %s dataset!' % (FLAGS.tfrecord_filename)

if __name__ == "__main__":
    main()
