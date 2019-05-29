import os
path=r'E:\tsl_file\windows_v1.8.1\windows_v1.8.1\label_data_self\labelled'
path_images=r'E:\tsl_file\windows_v1.8.1\windows_v1.8.1\label_data_self\raw'
a=os.listdir(path)

path=r'..\raw'
# path=path_images
g = os.walk(path)
label_dict={}
counts=0

for path,dir_list,file_list in g:

    print('paths',path)
    print('dir_list',dir_list)
    print('dir_list', dir_list)
    for file_name in file_list:
        # print(os.path.join(path, file_name) )
        print(path,file_name)
#         label=path.split('\\')[-1]
#         if label not in label_dict:
#             counts=counts+1
#             label_dict[label]=counts
# print(len(label_dict),label_dict)
# label_dict={'alibaba': 1, 'baidu': 2, 'bjtv': 3, 'cctv': 4, 'emblem': 5, 'hntv': 6, 'huawei': 7, 'jstv': 8, 'lenovo': 9, 'lexus': 10, 'lincoln': 11, 'starbucks': 12, 'supor': 13, 'suzuki': 14, 'tcl': 15, 'tencent': 16, 'tesla': 17, 'toshiba': 18, 'toyota': 19, 'tsingdao': 20, 'vatti': 21, 'vivo': 22, 'volvo': 23, 'walmart': 24, 'wanda': 25, 'wuliangye': 26, 'xiaomi': 27, 'yonghui': 28, 'yuantong': 29, 'zjtv': 30}
