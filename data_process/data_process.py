import os
# path=r'E:\tsl_file\windows_v1.8.1\windows_v1.8.1\label_data_self\raw\baidu'
import os
import shutil
from random import shuffle

def rename_jpg():
    path=r'E:\tsl_file\windows_v1.8.1\windows_v1.8.1\label_data_self\labelled\alibaba'
    all_jpg_names=os.listdir(path)
    for each_jpg in all_jpg_names:
        abs_jpg_path=os.path.join(path,each_jpg)
        os.rename(abs_jpg_path,os.path.join(path,'alibaba'+each_jpg))

        # print(abs_jpg_path)
def merge_jpg():

    # print('输入格式：E:\myprojectnew\jupyter\整理文件夹\示例')
    # path = input('请键入需要整理的文件夹地址：')
    # new_path = input('请键入要复制到的文件夹地址：')

    path='../raw/'
    new_path='../test_images_new'


    for path,dir_list,file_list in os.walk(path):
        # print('path',path)
        # print('dir_list',dir_list)
        # print('file_list',file_list)

        '''
        path ../raw/
        dir_list ['alibaba', 'baidu', 'bjtv', 'cctv', 'emblem', 'hntv', 'huawei', 'jstv', 'lenovo', 'lexus', 'lincoln', 'starbucks', 'supor', 'suzuki', 'tcl', 'tencent', 'tesla', 'toshiba', 'toyota', 'tsingdao', 'vatti', 'vivo', 'volvo', 'walmart', 'wanda', 'wuliangye', 'xiaomi', 'yonghui', 'yuantong', 'zjtv']
        file_list []
        path ../raw/alibaba
        dir_list []
        file_list ['1.jpg', '10.jpg', '11.jpg', '12.jpg', '13.jpg', '14.jpg', '15.jpg', '16.jpg', '17.jpg', '2.jpg', '3.jpg', '4.jpg', '5.jpg', '6.jpg', '7.jpg', '8.jpg', '9.jpg']
        path ../raw/baidu
        dir_list []
        file_list ['1.jpg', '10.jpg', '11.jpg', '12.jpg', '13.jpg', '14.jpg', '15.jpg', '16.jpg', '2.jpg', '3.jpg', '4.jpg', '5.jpg', '6.jpg', '7.jpg', '8.jpg', '9.jpg']

        
        '''
        shuffle(file_list)#打乱文件名顺序
        for index, file_name in enumerate(file_list):
            if index>2:
                break
            extend_name=os.path.splitext(file_name)[1]
            print('extend_name',extend_name)
            if extend_name in ['.jpg','.png','.jpeg']:
                pathjpg=os.path.join(path+'/',file_name)

                new_pathjpg=os.path.join(new_path+'/',path.split('/')[-1]+file_name)

                print('pathjpg',pathjpg)
                print('new_pathjpg',new_pathjpg)
                shutil.copy(pathjpg, new_pathjpg)




    # for root, dirs, files in os.walk(path):
        # for i in range(len(files)):
        #     print(files[i])
            # if (files[i][-3:] == 'jpg') or (files[i][-3:] == 'png') or (files[i][-3:] == 'JPG'):
            #     file_path = root + '/' + files[i]
            #     new_file_path = new_path + '/' + files[i]
            #     shutil.copy(file_path, new_file_path)

                # yn_close = input('是否退出？')
if __name__=='__main__':
    # merge_jpg()

    # os.path.split(path) - ---- 分割路径名和文件名，返回值为tuple：(路径名，文件名.文件扩展名)，如果路径是目录，则返回：(路径名，'')
    # os.path.splitext(path) - ---- 分离文件名和扩展名，返回值格式：(文件名，扩展名)，如果参数是目录，则返回：(路径，'')
    # print(os.path.split('F://hh/gg.jpg'))
    # print(os.path.splitext('gg.jpg'))

    # data = [1, 2, 4, 5]
    # shuffle(data)
    # print(data)
    # [2, 3, 4, 5, 1]

    # print(os.listdir('../raw/alibaba'))
    '''['1.jpg', '10.jpg', '11.jpg', '12.jpg'''
