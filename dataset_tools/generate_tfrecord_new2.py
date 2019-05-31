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
flags.DEFINE_string('label_path', '../data_examples/label_dict.txt', 'path to class label')
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


def split(df, group0,group1):
    data = namedtuple('data', ['filename', 'object'])
    gb = df.groupby([group0,group1])
    print('gb.groups',gb.groups)
    print('gb.groups.keys()',gb.groups.keys())
    each_class_all_jpgs=[]
    all_class_dict={"alibaba": [], "baidu": [], "bjtv": [], "cctv": [], "emblem": [], "hntv": [],
                    "huawei": [], "jstv": [], "lenovo": [], "lexus": [], "lincoln": [], "starbucks": [],
                    "supor": [], "suzuki": [], "tcl": [], "tencent": [], "tesla": [], "toshiba": [],
                    "toyota": [], "tsingdao": [], "vatti": [], "vivo": [], "volvo": [], "walmart": [],
                    "wanda": [], "wuliangye": [], "xiaomi": [], "yonghui": [], "yuantong": [], "zjtv": []}

    for filename, x in zip(gb.groups.keys(), gb.groups):
        # print('filename:',filename)
        # print('x:',x)##filename和x值是一样的，字典通过x获取值
        # print('gb.get_group(x)',gb.get_group(x))
        all_class_dict[filename[0]].append(data(filename, gb.get_group(x)))
    # print('all_class_dict',all_class_dict)
    return all_class_dict




    # return { filename[0]:data(filename, gb.get_group(x)) for filename, x in zip(gb.groups.keys(), gb.groups)}

    # return [data(filename, gb.get_group(x)) for filename, x in zip(gb.groups.keys(), gb.groups)]


def create_tf_example(group, path):
    new_path=os.path.join(path, '{}'.format(group.filename[1]))
    print('new_path',new_path)
    with tf.gfile.GFile(new_path, 'rb') as fid:
        ###通过csv文件获取图片路径并读取
        encoded_jpg = fid.read()
    encoded_jpg_io = io.BytesIO(encoded_jpg)
    image = Image.open(encoded_jpg_io)
    width, height = image.size

    filename = group.filename[0].encode('utf8')
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
        # print('row ',row )
        # print('classes', classes)
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
    # path = os.path.join(os.getcwd(), FLAGS.img_path)
    examples = pd.read_csv(FLAGS.csv_input)

    print('examples',examples.shape,examples.sample(50))


    #grouped以图片为单位，一张图片上的多个框坐标为一个grouped
    # grouped = split(examples, 'filename','class')
    grouped = split(examples, 'folder', 'filename')

    # print('grouped',grouped)

    g = os.walk(FLAGS.img_path)
    for paths, dir_list, file_list in g:

        folder_jpg = paths.split('/')[-1]
        ##传过来jpg的folder要根据此folder寻找每个类别文件夹下的所有同类图片
        # print('path1',paths)
        if folder_jpg=='':
            print('path null',paths)
            continue
        else:
            for each_jpg in grouped[folder_jpg]:
                # print(os.path.join(path, file_name) )
                # print(path, file_name)
                # folder_jpg=paths.split('\\')[-1]
                print(' paths', paths)
                print('folder_jpg',folder_jpg)###类别名称
                print('each_jpg',each_jpg)
                #文件夹名称不一致造成;由于打完标签后原来的文件名称被修改了造成的，例如原来的名称是‘阿里巴巴’
                #打完标签后改为‘alibaba'，造成用改名的文件名作为键就找不到了；要使得xml里文件名夹称和jpg文件名称一样才可以
            ###这里为了能运行改了，xmL的文件夹名称和jpg文件夹名称一致
                #将阿里巴巴改为alibaba;t_全部去掉就和jpg文件夹一致了
                # print('group',grouped[folder_jpg])
                #paths是images根目录，组合csv的图片文件名字，以此读取相应的文件
                ##应该用folder作为键

                ##传进来每一张图片及其所有的labels作为一个example
                tf_example = create_tf_example(each_jpg, paths)  ###只要group和path对应即可
                writer.write(tf_example.SerializeToString())


    writer.close()
    output_path = os.path.join(os.getcwd(), FLAGS.output_path)
    print('Successfully created the TFRecords: {}'.format(output_path))


if __name__ == '__main__':
    tf.app.run()


    #####下面是自己测试使用的
    # examples = pd.read_csv('../data_examples/train_labels_new.csv')
    #
    # print('examples', examples.shape, examples.sample(50))
    #
    # # grouped以图片为单位，一张图片上的多个框坐标为一个grouped
    # # grouped = split(examples, 'filename','class')
    # grouped = split(examples, 'folder', 'filename')
    # print('grouped',grouped)