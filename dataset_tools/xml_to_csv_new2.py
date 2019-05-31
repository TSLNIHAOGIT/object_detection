"""
Usage:
# Create train data:
python xml_to_csv.py -i [PATH_TO_IMAGES_FOLDER]/train -o [PATH_TO_ANNOTATIONS_FOLDER]/train_labels.csv

# Create test data:
python xml_to_csv.py -i [PATH_TO_IMAGES_FOLDER]/test -o [PATH_TO_ANNOTATIONS_FOLDER]/test_labels.csv
"""

import os
import glob
import pandas as pd
import argparse
import xml.etree.ElementTree as ET


def all_folder_files(rootpath):
    xml_list = []
    g = os.walk(rootpath)
    for path,dir_list,file_list in g:
        # for file_name in file_list:
        #     # print(os.path.join(path, file_name) )
        #     print(path,file_name)

            #path是文件夹，只有里面有xml文件就会一次性取出来
            for xml_file in glob.glob(path + '/*.xml'):
                tree = ET.parse(xml_file)
                root = tree.getroot()
                for member in root.findall('object'):
                    value = (
                            root.find('folder').text,
                            root.find('filename').text,
                            int(root.find('size')[0].text),
                            int(root.find('size')[1].text),
                            member[0].text,
                            int(member[4][0].text),
                            int(member[4][1].text),
                            int(member[4][2].text),
                            int(member[4][3].text)
                            )
                    xml_list.append(value)
    column_name = ['folder','filename', 'width', 'height',
                'class', 'xmin', 'ymin', 'xmax', 'ymax']
    xml_df = pd.DataFrame(xml_list, columns=column_name)
    return xml_df


def main():
    # Initiate argument parser
    parser = argparse.ArgumentParser(
        description="Sample TensorFlow XML-to-CSV converter")
    parser.add_argument("-i",
                        "--inputDir",
                        help="Path to the folder where the input .xml files are stored",
                        type=str)
    parser.add_argument("-o",
                        "--outputFile",
                        help="Name of output .csv file (including path)", type=str)
    args = parser.parse_args()

    print('args.inputDir',args.inputDir)

    # if(args.inputDir is None):
    #     args.inputDir = os.getcwd()
    # if(args.outputFile is None):
    #     args.outputFile = args.inputDir + "/labels.csv"

    # assert(os.path.isdir(args.inputDir))

    # xml_df = xml_to_csv(args.inputDir)
    # args.inputDir='../data_samples_new'
    # args.inputDir=r'E:/tsl_file/windows_v1.8.1/windows_v1.8.1/label_data_self/labelled'
    xml_df=all_folder_files(args.inputDir)
    print('xml_df',xml_df.head())
    xml_df.to_csv(
        args.outputFile, index=None)
    print('Successfully converted xml to csv.')


if __name__ == '__main__':
    main()