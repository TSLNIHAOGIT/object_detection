"""
Usage:

# Create train data:
python generate_tfrecord.py --label=<LABEL> --csv_input=<PATH_TO_ANNOTATIONS_FOLDER>/train_labels.csv  --output_path=<PATH_TO_ANNOTATIONS_FOLDER>/train.record

# Create test data:
python generate_tfrecord.py --label=<LABEL> --csv_input=<PATH_TO_ANNOTATIONS_FOLDER>/test_labels.csv  --output_path=<PATH_TO_ANNOTATIONS_FOLDER>/test.record
"""

from __future__ import division
from __future__ import print_function
from __future__ import absolute_import

import os
import io
import pandas as pd
import tensorflow as tf
import sys
# sys.path.append("../../models/research")

# sys.path.append("..")
import json
#要把object_detection的上级目录加到搜索路径中，才能搜到object_detection模块
path=os.path.abspath(os.path.join(os.path.dirname(__file__),'../..'))

print('path',path)
sys.path.insert(0,path)
# print(sys.path)





###下面这个方法也可以
# curPath = os.path.abspath(os.path.dirname(__file__))
# print('curPath',curPath)
# rootPath = os.path.split(curPath)[0]
# print('rootPath',rootPath)
# sys.path.append(os.path.split(rootPath)[0])
# print('mypath',os.path.split(rootPath)[0])





from PIL import Image
from object_detection.utils import dataset_util
from collections import namedtuple, OrderedDict

flags = tf.app.flags
flags.DEFINE_string('csv_input', '', 'Path to the CSV input')
flags.DEFINE_string('output_path', '', 'Path to output TFRecord')
flags.DEFINE_string('label_path', '', 'path to class label')
# if your image has more labels input them as
# flags.DEFINE_string('label0', '', 'Name of class[0] label')
# flags.DEFINE_string('label1', '', 'Name of class[1] label')
# and so on.
flags.DEFINE_string('img_path', '', 'Path to images')
FLAGS = flags.FLAGS

with open(FLAGS.label_path) as f:
    label_dict=json.load(f)

# TO-DO replace this with label map
# for multiple labels add more else if statements
def class_text_to_int(row_label):
    # if row_label == FLAGS.label:  # 'ship':
    #     return 1
    # # comment upper if statement and uncomment these statements for multiple labelling
    # # if row_label == FLAGS.label0:
    # #   return 1
    # # elif row_label == FLAGS.label1:
    # #   return 0
    # else:
    #     None
    return label_dict[row_label]


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

    filename = group.filename.encode('utf8')
    image_format = b'jpg'
    # check if the image format is matching with your images.
    xmins = []
    xmaxs = []
    ymins = []
    ymaxs = []
    classes_text = []
    classes = []

    for index, row in group.object.iterrows():

        xmins.append(row['xmin'] / width)
        xmaxs.append(row['xmax'] / width)
        ymins.append(row['ymin'] / height)
        ymaxs.append(row['ymax'] / height)
        classes_text.append(row['class'].encode('utf8'))
        classes.append(class_text_to_int(row['class']))
        print('row ',row )
        print('classes', classes)
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
    path = os.path.join(os.getcwd(), FLAGS.img_path)
    examples = pd.read_csv(FLAGS.csv_input)
    grouped = split(examples, 'filename')
    for group in grouped:
        tf_example = create_tf_example(group, path)
        # print('group',group)
        # print( 'path', path)
        writer.write(tf_example.SerializeToString())

    writer.close()
    output_path = os.path.join(os.getcwd(), FLAGS.output_path)
    print('Successfully created the TFRecords: {}'.format(output_path))


if __name__ == '__main__':
    tf.app.run()

#一个图片多个label为一个group
'''
path F:\陶士来文件\tsl_python_project
group data(filename='1.jpg', object=  filename  width  height    class  xmin  ymin  xmax  ymax
0    1.jpg    500     318  alibaba   186   214   328   247
1    1.jpg    500     318  alibaba   187   249   328   278)
path F:\陶士来文件\tsl_python_project\object_detection\dataset_tools\../data_examples/images
group data(filename='2.jpg', object=  filename  width  height    class  xmin  ymin  xmax  ymax
2    2.jpg    500     375  alibaba   154   199   500   328)
path F:\陶士来文件\tsl_python_project\object_detection\dataset_tools\../data_examples/images
group data(filename='3.jpg', object=  filename  width  height    class  xmin  ymin  xmax  ymax
3    3.jpg    500     334  alibaba     1    20   487   297)
path F:\陶士来文件\tsl_python_project\object_detection\dataset_tools\../data_examples/images
group data(filename='4.jpg', object=  filename  width  height    class  xmin  ymin  xmax  ymax
4    4.jpg    500     334  alibaba     8   138   490   226)
path F:\陶士来文件\tsl_python_project\object_detection\dataset_tools\../data_examples/images
group data(filename='5.jpg', object=  filename  width  height    class  xmin  ymin  xmax  ymax
5    5.jpg    499     339  alibaba     1     1   499   243)
path F:\陶士来文件\tsl_python_project\object_detection\dataset_tools\../data_examples/images
'''