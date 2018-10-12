"""
$ python csv_data_parser.py --normalized(default=False)

1. According to 'classes-trainable.csv', she filters out untrainnalbe labels in 'train_human_labels.csv' and 'train_machine_labels.csv', and save them as 'fixed_train_human_labels.csv' and 'fixed_train_machine_labels.csv'.
3. Visulize the bar chart of labels in 'tuning_labels.csv', 'fixed_train_human_labels.csv', and 'fixed_train_machine_labels.csv'.
"""
import csv
import os
import copy
import operator
import shutil
import plotly
import plotly.graph_objs as go
import argparse

IMAGES_PATH = '../inputs'
LABELS_PATH = '../labels'


def filter_out_untrainable_label():
    """ According to 'classes-trainable.csv', she filters out untrainnalbe labels in 'train_human_labels.csv' and 'train_machine_labels.csv', and save them as 'fixed_train_human_labels.csv' and 'fixed_train_machine_labels.csv'.
    """
    trainable_label_filepath = os.path.join(
        LABELS_PATH, 'classes-trainable.csv')
    # ['labelcode']
    cond_reader = csv.reader(
        open(trainable_label_filepath, 'r'), delimiter=',')
    trainable_list = []
    for row in cond_reader:
        trainable_list.append(row[0])
    for csv_filename in ['train_human_labels.csv', 'train_machine_labels.csv']:
        csv_filepath = os.path.join(LABELS_PATH, csv_filename)
        out_filepath = os.path.join(LABELS_PATH, 'fixed_'+csv_filename)
        # ['ImageID', 'Source', 'LabelName', 'Confidence']
        file_reader = csv.reader(open(csv_filepath, 'r'), delimiter=',')

        global counter
        counter = 0

        def judger_fn(row):
            global counter
            counter = counter + 1
            if counter % 100000 == 0:
                print(counter)
            return row[2] in trainable_list

        label_filter = filter(judger_fn, file_reader)
        # label_filter = filter(lambda p: p[2] in trainable_list, file_reader)
        file_writer = csv.writer(open(out_filepath, 'w'), delimiter=',')
        file_writer.writerows(label_filter)
        print(file_reader.line_num)


def vis_barchart(normalized=False):
    """ Visulize the histogram of labels in 'tuning_labels.csv', 'fixed_train_human_labels.csv', and 'fixed_train_machine_labels.csv'.
    """
    trainable_label_filepath = os.path.join(
        LABELS_PATH, 'classes-trainable.csv')
    cond_reader = csv.reader(
        open(trainable_label_filepath, 'r'), delimiter=',')
    trainable_dict = {}
    for i, row in enumerate(cond_reader):
        if i != 0:  # first row is ['labelcode']
            trainable_dict[row[0]] = 0

    tuning_label_filepath = os.path.join(
        LABELS_PATH, 'tuning_labels.csv')
    human_label_filepath = os.path.join(
        LABELS_PATH, 'fixed_train_human_labels.csv')
    machine_label_filepath = os.path.join(
        LABELS_PATH, 'fixed_train_machine_labels.csv')
    classes_name_filepath = os.path.join(
        LABELS_PATH, 'class-descriptions.csv')

    # ['label_code', 'description']
    file_reader = csv.reader(
        open(classes_name_filepath, 'r', encoding='utf-8'), delimiter=',')
    class_mapping = {}
    for i, row in enumerate(file_reader):
        if i != 0:
            class_mapping[row[0]] = row[1]

    ########################################################################
    # human_label
    ########################################################################
    # use human_label to get order or x-axis in bar chart
    counter_dict = copy.deepcopy(trainable_dict)
    file_reader = csv.reader(open(human_label_filepath, 'r'), delimiter=',')
    for i, row in enumerate(file_reader):
        if i != 0:  # first row is ['ImageID', 'Source', 'LabelName', 'Confidence']
            counter_dict[row[2]] = counter_dict[row[2]] + 1
    # sort dict by transforming it into list
    human_sorted_tuple_list = sorted(counter_dict.items(
    ), key=operator.itemgetter(1), reverse=True)  # sort dict by value
    if normalized:
        total_counts = sum(list(counter_dict.values()))
    else:
        total_counts = 1
    trace1 = go.Bar(
        x=[class_mapping[t[0]] for t in human_sorted_tuple_list],
        y=[t[1]/total_counts for t in human_sorted_tuple_list],
        name='Human Labels',
        opacity=0.6
    )

    ########################################################################
    # machine_label
    ########################################################################
    counter_dict = copy.deepcopy(trainable_dict)
    file_reader = csv.reader(open(machine_label_filepath, 'r'), delimiter=',')
    for i, row in enumerate(file_reader):
        if i != 0:  # first row is ['ImageID', 'Source', 'LabelName', 'Confidence']
            counter_dict[row[2]] = counter_dict[row[2]] + 1
    key_order = [t[0] for t in human_sorted_tuple_list]
    if normalized:
        total_counts = sum(list(counter_dict.values()))
    else:
        total_counts = 1
    trace2 = go.Bar(
        x=[class_mapping[t[0]] for t in human_sorted_tuple_list],
        y=[counter_dict[key]/total_counts for key in key_order],
        name='Machine Labels',
        opacity=0.6
    )

    ########################################################################
    # tuning_label
    ########################################################################
    counter_dict = copy.deepcopy(trainable_dict)
    file_reader = csv.reader(open(tuning_label_filepath, 'r'), delimiter=',')
    not_found_counter = 0
    for i, row in enumerate(file_reader):
        if i != 0:  # first row is ['ImageID', 'LabelNamesss']
            # copy images (optional)
            test_image_path = os.path.join(IMAGES_PATH, 'stage_1_test_images')
            validation_image_path = os.path.join(IMAGES_PATH, 'stage_1_validation_images')
            os.makedirs(validation_image_path, exist_ok=True)
            shutil.copyfile(os.path.join(test_image_path, str(row[0])+'.jpg'),
                    os.path.join(validation_image_path, str(row[0])+'.jpg'))
            # tally
            labels = str(row[1]).split(' ')
            for label in labels:
                counter_dict[label] = counter_dict[label] + 1
    key_order = [t[0] for t in human_sorted_tuple_list]
    if normalized:
        total_counts = sum(list(counter_dict.values()))
    else:
        total_counts = 1
    trace3 = go.Bar(
        x=[class_mapping[t[0]] for t in human_sorted_tuple_list],
        y=[counter_dict[key]/total_counts for key in key_order],
        name='Tuning Labels',
        opacity=0.6
    )

    data = [trace1, trace2, trace3]
    layout = go.Layout(
        barmode='group'
    )

    fig = go.Figure(data=data, layout=layout)
    if normalized:
        plotly.offline.plot(fig, filename='(normalized)labels_bar_chart.html')
    else:
        plotly.offline.plot(fig, filename='(counts)labels_bar_chart.html')


def main():
    parser = argparse.ArgumentParser(description='')
    parser.add_argument('--normalized', default=False,
                        help='normalize the counts of bar chart')
    args = parser.parse_args()
    # filter_out_untrainable_label()
    vis_barchart(normalized=args.normalized)


if __name__ == '__main__':
    main()
