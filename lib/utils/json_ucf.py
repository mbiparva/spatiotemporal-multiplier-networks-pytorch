import os
import sys
import json
import pandas as pd


def load_labels(label_path):
    label_content = pd.read_csv(label_path, sep='delimiter', header=None, engine='python')
    labels = {}
    for i in range(len(label_content)):
        cls_id, cls_name = label_content.iloc[i, 0].split(' ')
        labels[int(cls_id)] = cls_name
    return labels


def csv_to_json(path, mode, labels):
    images_path = os.path.normpath(os.path.join(path, '../../../images'))
    label_values = list(labels.values())
    csv_content = pd.read_csv(path, header=None)
    class_list, video_list, nframes_list = [], [], []
    for i in range(len(csv_content)):
        row = csv_content.iloc[i, 0]
        row_partitioned = row.split('/')
        class_list.append(row_partitioned[0])
        video_list.append(row_partitioned[1].split('.')[0])
        video_path = os.path.join(images_path, video_list[-1])
        nframes_list.append(len(os.listdir(video_path)))
        print(i, row, 'done!')
    annotation_dict = {}
    for i in range(len(video_list)):
        annotation_dict[video_list[i]] = {
            'set':  mode,
            'label': label_values.index(class_list[i]) + 1,   # 0 is the background
            'nframes': nframes_list[i],
        }

    return annotation_dict


def csv_to_json_container(label_path, t, v, a):
    labels = load_labels(label_path)
    train_dict = csv_to_json(t, 'training', labels)
    valid_dict = csv_to_json(v, 'validation', labels)

    dataset_dict = {
        'labels': labels,
        'training': train_dict,
        'validation': valid_dict
    }

    with open(a, 'w') as json_file_handle:
        json.dump(dataset_dict, json_file_handle)


if __name__ == '__main__':
    annotation_path = sys.argv[1]

    for split_idx in range(1, 4):
        label_p = os.path.join(annotation_path, 'classInd.txt')
        t_p = os.path.join(annotation_path, 'trainlist0{}.txt'.format(split_idx))
        v_p = os.path.join(annotation_path, 'testlist0{}.txt'.format(split_idx))
        a_p = os.path.join(annotation_path, 'annot0{}.json'.format(split_idx))

        csv_to_json_container(label_p, t_p, v_p, a_p)
