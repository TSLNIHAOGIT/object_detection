"""
#TO-DO replace this with label map下的内容需要根据自己的数据集进行修改，修改为自己数据集的标签。
Usage:
  # From tensorflow/models/
  # Create train data:
  python generate_tfrecord.py --csv_input=data/train_labels.csv  --output_path=train.record
  # Create test data:
  python generate_tfrecord.py --csv_input=data/test_labels.csv  --output_path=test.record
"""
from __future__ import division
from __future__ import print_function
from __future__ import absolute_import

import os,sys
import io
import pandas as pd
import tensorflow as tf
import json

from PIL import Image
path=os.path.abspath(os.path.join(os.path.dirname(__file__),'..'))
print('path',path)
sys.path.insert(0,path)
print(sys.path)
from utils import dataset_util
from collections import namedtuple, OrderedDict

flags = tf.app.flags
flags.DEFINE_string('csv_input', '../data_examples/data_samples_labels.csv', 'Path to the CSV input')
flags.DEFINE_string('output_path', '../data_tfrecord_samples/out.record', 'Path to output TFRecord')
FLAGS = flags.FLAGS


# TO-DO replace this with label map
def class_text_to_int(row_label):
    #注意这里的label一定要和xml中的一样，否则会报错（为空，即没找到这个label）
    if row_label == 'aliw':
        return 1
    elif row_label == 'baiduww':
        return 2
    else:
        None


def split(df, group):
    data = namedtuple('data', ['filename', 'object'])
    gb = df.groupby(group)
    return [data(filename, gb.get_group(x)) for filename, x in zip(gb.groups.keys(), gb.groups)]


def create_tf_example(group, path):
    with tf.gfile.GFile(os.path.join(path, '{}'.format(group.filename)), 'rb') as fid:
        encoded_jpg = fid.read()
    encoded_jpg_io = io.BytesIO(encoded_jpg)
    image = Image.open(encoded_jpg_io)
    width, height = image.size
    # print('width, height',width, height)

    filename = group.filename.encode('utf8')
    image_format = b'jpg'
    xmins = []
    xmaxs = []
    ymins = []
    ymaxs = []
    classes_text = []
    classes = []

    for index, row in group.object.iterrows():
        print('row',row)
        xmins.append(row['xmin'] / width)
        xmaxs.append(row['xmax'] / width)
        ymins.append(row['ymin'] / height)
        ymaxs.append(row['ymax'] / height)
        classes_text.append(row['class'].encode('utf8'))

        classes.append(class_text_to_int(row['class']))
    print('classes',classes)
    tf_example = tf.train.Example(features=tf.train.Features(feature={
        'image/height': dataset_util.int64_feature(height),
        'image/width': dataset_util.int64_feature(width),
        'image/filename': dataset_util.bytes_feature(filename),
        'image/source_id': dataset_util.bytes_feature(filename),
        'image/encoded': dataset_util.bytes_feature(encoded_jpg),
        'image/format': dataset_util.bytes_feature(image_format),
        'image/object/bbox/xmin': dataset_util.float_list_feature(xmins),
        'image/object/bbox/xmax': dataset_util.float_list_feature(xmaxs),
        'image/object/bbox/ymin': dataset_util.float_list_feature(ymins),
        'image/object/bbox/ymax': dataset_util.float_list_feature(ymaxs),
        'image/object/class/text': dataset_util.bytes_list_feature(classes_text),
        'image/object/class/label': dataset_util.int64_list_feature(classes),
    }))
    return tf_example



def main(_):
    writer = tf.python_io.TFRecordWriter(FLAGS.output_path)
    # path = os.path.join(os.getcwd(), 'images')
    path='../data_examples/images'
    examples = pd.read_csv(FLAGS.csv_input)
    grouped = split(examples, 'filename')
    # print('grouped ',grouped )
    for group in grouped:
        print('group',group)
        tf_example = create_tf_example(group, path)
        writer.write(tf_example.SerializeToString())
        '''
        group data(filename='ali2.jpg', object=   filename  width  height class  xmin  ymin  xmax  ymax
        0  ali2.jpg    293     220  aliw    43    88   240   138)
        group data(filename='ali3.jpg', object=   filename  width  height class  xmin  ymin  xmax  ymax
        1  ali3.jpg    318     220  aliw    37    61   258   152)
        '''

    writer.close()
    output_path = os.path.join(os.getcwd(), FLAGS.output_path)
    print('Successfully created the TFRecords: {}'.format(output_path))


if __name__ == '__main__':
    tf.app.run()
