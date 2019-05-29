import json
path='../data_examples/label_dict.txt'
save_path='../data/train_label_map.pbtxt'
with open(path) as f:
    label_dict=json.load(f)
    print(label_dict)
    all_map=[]
    with open(save_path,mode='a+') as fw:
        for each in label_dict:
            fw.write('item {\n')
            fw.write('  id:{}\n'.format(label_dict[each]))
            fw.write("  name:'{}'\n".format(each))
            fw.write('}\n\n')

            # item={}
            # item['id']=label_dict[each]
            # item['name']=each
            # all_map.append(item)
# print(all_map)
