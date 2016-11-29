from datetime import datetime
import argparse
import os
import random
import sys
import threading

import numpy as np
import tensorflow as tf

tf.app.flags.DEFINE_string('train', 'train', 'Training images directory')
tf.app.flags.DEFINE_string('validation', 'validation', 'Validation images directory')
tf.app.flags.DEFINE_string('train_labels', 'train-labels', 'Training label images directory')
tf.app.flags.DEFINE_string('validation_labels', 'validation-labels', 'Validation label images directory')
tf.app.flags.DEFINE_string('labels_file', 'labels', 'Labels file')
tf.app.flags.DEFINE_string('output', 'output', 'Output data directory')

tf.app.flags.DEFINE_boolean('encode_validation_images', False, 'Encode validation images')
tf.app.flags.DEFINE_boolean('encode_labels', False, 'Encode labels as images')
tf.app.flags.DEFINE_boolean('resize', True, 'Resize to fit image_side_length as lower side')
tf.app.flags.DEFINE_boolean('square', True, 'Crop to get square target images')
tf.app.flags.DEFINE_boolean('shuffle', True, 'Shuffle images')

tf.app.flags.DEFINE_integer('image_side_length', 224, 'Length of the side of the square target images')

tf.app.flags.DEFINE_integer('train_shards', 1, 'Number of shards in training TFRecord files')
tf.app.flags.DEFINE_integer('validation_shards', 1, 'Number of shards in validation TFRecord files')
tf.app.flags.DEFINE_integer('threads', 1, 'Number of threads to preprocess the images')

FLAGS = tf.app.flags.FLAGS


def _int64_feature(value):
  if not isinstance(value, list):
    value = [value]
  return tf.train.Feature(int64_list=tf.train.Int64List(value=value))


def _bytes_feature(value):
  return tf.train.Feature(bytes_list=tf.train.BytesList(value=[value]))


def _convert_to_example(filename, image_buffer, label, text, height, width):
  colorspace = str.encode('RGB')
  text = str.encode(text)
  channels = 3
  image_format = str.encode('JPEG')
  filename = str.encode(os.path.basename(filename))

  if FLAGS.encode_labels:
    example = tf.train.Example(features=tf.train.Features(feature={
      'image/encoded': _bytes_feature(image_buffer)
    }))
  else:
    example = tf.train.Example(features=tf.train.Features(feature={
      'image/class/label': _int64_feature(label),
      'image/encoded': _bytes_feature(image_buffer)
    }))
  return example


class ImageCoder(object):
  def __init__(self):
    self._sess = tf.Session()

    self._png_data = tf.placeholder(dtype=tf.string)
    self._decode_png = tf.image.decode_png(self._png_data, channels=3)
    self._png_to_jpeg = tf.image.encode_jpeg(self._decode_png, format='rgb', quality=100)

    self._decode_jpeg_data = tf.placeholder(dtype=tf.string)
    self._decode_jpeg = tf.image.decode_jpeg(self._decode_jpeg_data, channels=3)
    self._encode_jpeg = tf.image.encode_jpeg(self._decode_jpeg, format='rgb', quality=100)

    self._height = tf.placeholder(dtype=tf.int32)
    self._width = tf.placeholder(dtype=tf.int32)
    self._real_height = tf.placeholder(dtype=tf.int32)
    self._real_width = tf.placeholder(dtype=tf.int32)
    self._resize = tf.image.resize_images(self._decode_jpeg, [self._height, self._width])
    self._resize_and_crop = tf.image.resize_image_with_crop_or_pad(self._resize, FLAGS.image_side_length, FLAGS.image_side_length)
    self._encode_resized_jpeg = tf.image.encode_jpeg(tf.cast(self._resize_and_crop, tf.uint8), format='rgb', quality=100)

  def png_to_jpeg(self, image_data):
    return self._sess.run(self._png_to_jpeg, feed_dict={self._png_data: image_data})

  def decode_jpeg(self, image_data):
    image = self._sess.run(self._decode_jpeg, feed_dict={self._decode_jpeg_data: image_data})
    assert len(image.shape) == 3
    assert image.shape[2] == 3
    return image

  def encode_jpeg(self, image_data):
    return self._sess.run(self._encode_jpeg, feed_dict={self._decode_jpeg_data: image_data})

  def resize_and_crop(self, image_data, height, width, real_height, real_width):
    return self._sess.run(self._encode_resized_jpeg, feed_dict={
      self._decode_jpeg_data: image_data,
      self._height: height,
      self._width: width,
      self._real_height: real_height,
      self._real_width: real_width
    })


def _is_png(filename):
  return '.png' in filename


def _resize(image):
  base = FLAGS.image_side_length
  height, width, _ = image.shape

  if width <= height:
    scaling_factor = (base / float(width))
    other = int(height * scaling_factor)
    shape = (other, base)
  else:
    scaling_factor = (base / float(height))
    other = int(width * scaling_factor)
    shape = (base, other)

  return shape


def _process_image(filename, coder):
  with tf.gfile.FastGFile(filename, 'r') as f:
    image_data = f.read()

  try:
    if _is_png(filename):
      print('Converting PNG to JPEG for %s' % filename)
      image_data = coder.png_to_jpeg(image_data)
    else:
      image_data = coder.encode_jpeg(image_data)

    image = coder.decode_jpeg(image_data)
  except:
    print('Skipped %s' % filename)
    print('Maybe %s is not encoded as JPG or PNG?' % filename)

  if FLAGS.resize and FLAGS.square:
    real_height, real_width, _ = image.shape
    height, width = _resize(image)
    image_data = coder.resize_and_crop(image_data, height, width, real_height, real_width)
    image = coder.decode_jpeg(image_data)

  assert len(image.shape) == 3
  final_height, final_width, _ = image.shape
  assert final_height == final_width
  assert final_height == FLAGS.image_side_length
  assert image.shape[2] == 3

  return image_data, final_height, final_width


def _process_image_files_batch(coder, thread_index, ranges, name, filenames, texts, labels, num_shards):
  # Each thread produces N shards where N = int(num_shards / num_threads).
  # For instance, if num_shards = 128, and the num_threads = 2, then the first
  # thread would produce shards [0, 64).
  num_threads = len(ranges)
  assert not num_shards % num_threads
  num_shards_per_batch = int(num_shards / num_threads)

  shard_ranges = np.linspace(ranges[thread_index][0],
                             ranges[thread_index][1],
                             num_shards_per_batch + 1).astype(int)
  num_files_in_thread = ranges[thread_index][1] - ranges[thread_index][0]

  counter = 0
  for s in range(num_shards_per_batch):
    # Generate a sharded version of the file name, e.g. 'train-00002-of-00010'
    shard = thread_index * num_shards_per_batch + s
    output_filename = '%s-%.5d-of-%.5d' % (name, shard, num_shards)
    output_file = os.path.join(FLAGS.output, output_filename)
    writer = tf.python_io.TFRecordWriter(output_file)

    shard_counter = 0
    files_in_shard = np.arange(shard_ranges[s], shard_ranges[s + 1], dtype=int)
    for i in files_in_shard:
      filename = filenames[i]
      label = labels[i]
      text = texts[i]

      image_buffer, height, width = _process_image(filename, coder)

      example = _convert_to_example(filename, image_buffer, label, text, height, width)
      writer.write(example.SerializeToString())
      shard_counter += 1
      counter += 1

      if not counter % 1000:
        print('%s [thread %d]: Processed %d of %d images in thread batch.' %
              (datetime.now(), thread_index, counter, num_files_in_thread))
        sys.stdout.flush()

    writer.close()
    print('%s [thread %d]: Wrote %d images to %s' %
          (datetime.now(), thread_index, shard_counter, output_file))
    sys.stdout.flush()
    shard_counter = 0
  print('%s [thread %d]: Wrote %d images to %d shards.' %
        (datetime.now(), thread_index, counter, num_files_in_thread))
  sys.stdout.flush()


def _process_image_files(name, filenames, texts, labels, num_shards):
  assert len(filenames) == len(texts)
  assert len(filenames) == len(labels)

  # Break all images into batches with a [ranges[i][0], ranges[i][1]].
  spacing = np.linspace(0, len(filenames), FLAGS.threads + 1).astype(np.int)
  ranges = []
  for i in range(len(spacing) - 1):
    ranges.append([spacing[i], spacing[i+1]])

  # Launch a thread for each batch.
  print('Launching %d threads for spacings: %s' % (FLAGS.threads, ranges))
  sys.stdout.flush()

  # Create a mechanism for monitoring when all threads are finished.
  coord = tf.train.Coordinator()

  # Create a generic TensorFlow-based utility for converting all image codings.
  coder = ImageCoder()

  threads = []
  for thread_index in range(len(ranges)):
    args = (coder, thread_index, ranges, name, filenames,
            texts, labels, num_shards)
    t = threading.Thread(target=_process_image_files_batch, args=args)
    t.start()
    threads.append(t)

  # Wait for all the threads to terminate.
  coord.join(threads)
  print('%s: Finished writing all %d images in data set.' %
        (datetime.now(), len(filenames)))
  sys.stdout.flush()


def _find_image_files(data_dir, labels_file):
  print('Determining list of input files and labels from %s.' % data_dir)
  unique_labels = [l.strip() for l in tf.gfile.FastGFile(
      labels_file, 'r').readlines()]

  labels = []
  filenames = []
  texts = []

  # Leave label index 0 empty as a background class.
  label_index = 1

  # Construct the list of JPEG files and labels.
  for text in unique_labels:
    jpeg_file_path = '%s/%s/*' % (data_dir, text)
    matching_files = tf.gfile.Glob(jpeg_file_path)

    labels.extend([label_index] * len(matching_files))
    texts.extend([text] * len(matching_files))
    filenames.extend(matching_files)

    if not label_index % 100:
      print('Finished finding files in %d of %d classes.' % (
          label_index, len(labels)))
    label_index += 1

  # Shuffle the ordering of all image files in order to guarantee
  # random ordering of the images with respect to label in the
  # saved TFRecord files. Make the randomization repeatable.
  if FLAGS.shuffle:
    shuffled_index = list(range(len(filenames)))
    random.seed(12345)
    random.shuffle(shuffled_index)

    filenames = [filenames[i] for i in shuffled_index]
    texts = [texts[i] for i in shuffled_index]
    labels = [labels[i] for i in shuffled_index]

  print('Found %d JPEG files across %d labels inside %s.' %
        (len(filenames), len(unique_labels), data_dir))
  return filenames, texts, labels


def _process_dataset(name, directory, num_shards, labels_file):
  filenames, texts, labels = _find_image_files(directory, labels_file)
  _process_image_files(name, filenames, texts, labels, num_shards)


def main(unused_argv):
  assert not FLAGS.train_shards % FLAGS.threads, ('Please make the FLAGS.threads commensurate with FLAGS.train_shards')
  assert not FLAGS.validation_shards % FLAGS.threads, ('Please make the FLAGS.threads commensurate with FLAGS.validation_shards')
  print('Saving results to %s' % FLAGS.output)

  _process_dataset('train', FLAGS.train, FLAGS.train_shards, FLAGS.labels_file)
  if FLAGS.encode_validation_images:
    _process_dataset('validation', FLAGS.validation, FLAGS.validation_shards, FLAGS.labels_file)
  if FLAGS.encode_labels:
    _process_dataset('train_labels', FLAGS.train_labels, FLAGS.train_shards, FLAGS.labels_file)
    if FLAGS.encode_validation_images:
      _process_dataset('validation_labels', FLAGS.validation_labels, FLAGS.validation_shards, FLAGS.labels_file)

if __name__ == '__main__':
  tf.app.run()
