import math
import copy


def if_no_blank(temp_list):
    flag = True
    for i in temp_list:
        if '__' in i:
            flag = False
            return flag
    return flag


def fill_ary(fLines, fAry, size):
    i = 0
    for line in fLines:
        if size == 0:
            return
        line = line.rstrip(line[-1])
        tempList = line.split(' ')
        if int(tempList[1]) + 2 > len(tempList)-1 or if_no_blank(tempList):
            continue
        fAry.insert(i, tempList)
        i += 1
        size -= 1


def calculate_total_entropy(total_count, train_data):
    count_whether, count_weather = 0, 0
    for x in range(total_count):
        if train_data[x][0] == 'whether':
            count_whether += 1
            continue
        else:
            count_weather += 1
            continue
    if count_weather <= 0 or count_whether <= 0:
        return 0
    total_entropy = -(count_weather/total_count)*math.log2(count_weather/total_count)-(count_whether/total_count)*math.log2(count_whether/total_count)
    return total_entropy


# Used Wikipedia Search: 'Confusion Matrix' to help with accuracy, recall, and precision formulas (Thought recall and precision was required :/)
# https://en.wikipedia.org/wiki/Confusion_matrix
def calculate_percentage(actual, predicted):
    TP, correct, FN, FP, TN = 0, 0, 0, 0, 0
    for i in range(len(actual)):
        if actual[i] == predicted[i]:
            correct += 1
            TP += 1
        if actual[i] == 'weather':
            FN += 1
        if predicted[i] == 'weather':
            FP += 1
        if actual[i] == 'whether' and predicted[i] == 'whether':
            TN += 1
    accuracy = correct / len(actual) * 100.0
    recall = TP / (TP + FN) * 100.0
    precision = TP / (TP + FP) * 100.0
    return accuracy, recall, precision


def calculate_information_gain(groups):
    total_group = groups[0] + groups[1]

    left_entropy = calculate_total_entropy(len(groups[0]), groups[0])
    right_entropy = calculate_total_entropy(len(groups[1]), groups[1])
    total_entropy = calculate_total_entropy(len(total_group), total_group)

    feature_value_probability1 = len(groups[0])/len(total_group)
    feature_value_probability2 = len(groups[1])/len(total_group)

    feature_info = (feature_value_probability1*left_entropy) + (feature_value_probability2*right_entropy)
    info_gain = total_entropy - feature_info
    return info_gain


def best_split(train_data, features):
    max_info_gain = -9999
    for x in features:
        children = [[], []]
        for line in train_data:
            if x in line:
                children[0].append(line)
            else:
                children[1].append(line)
        info_gain = calculate_information_gain(children)
        if info_gain > max_info_gain:
            max_info_gain = info_gain
            new_children = children
            feat = x
    return {'children': new_children, 'feat': feat}


def end_node_label(child):
    labels = [row[0] for row in child]
    count_label1, count_label2 = 0, 0
    for label in labels:
        if label == 'weather':
            count_label1 += 1
        else:
            count_label2 += 1
    if count_label1 > count_label2:
        return 'weather'
    else:
        return 'whether'


# base cases: if feature list is empty; if max_height of tree is reached; if no data left (leaf node)
# else split left child and right child
def recursive_split(tree_node, features, curr_height, max_height):
    features_left = copy.deepcopy(features)
    features_right = copy.deepcopy(features)
    curr_height_left = copy.deepcopy(curr_height)
    curr_height_right = copy.deepcopy(curr_height)
    left_child, right_child = tree_node['children']

    if len(features) == 0:
        tree_node['left_child'] = end_node_label(left_child)
        tree_node['right_child'] = end_node_label(right_child)
        return

    if max_height <= curr_height:
        tree_node['right_child'] = end_node_label(right_child)
        tree_node['left_child'] = end_node_label(left_child)
        return

    if len(right_child) == 0 or len(left_child) == 0:
        tree_node['right_child'] = tree_node['left_child'] = end_node_label(right_child + left_child)
        return

    if len(left_child) > 1:
        tree_node['left_child'] = best_split(left_child, features)
        features_left.remove(tree_node['left_child']['feat'])
        curr_height_left += 1
        recursive_split(tree_node['left_child'], features_left, curr_height_left, max_height)
    else:
        tree_node['left_child'] = end_node_label(left_child)

    if len(right_child) > 1:
        tree_node['right_child'] = best_split(right_child, features)
        features_right.remove(tree_node['right_child']['feat'])
        curr_height_right += 1
        recursive_split(tree_node['right_child'], features_right, curr_height_right, max_height)
    else:
        tree_node['right_child'] = end_node_label(right_child)


def build_decision_tree(train, features, max_height):
    root = best_split(train, features)
    features.remove(root['feat'])
    recursive_split(root, features, 1, max_height)
    return root


def decision_tree(train_data, test_data, features, max_height):
    tree = build_decision_tree(train_data, features, max_height)
    prediction_list = []
    for line in test_data:
        prediction = make_predictions(tree, line)
        prediction_list.append(prediction)
    return prediction_list


# search through tree until leaf node is reached, return label of tree_node
def make_predictions(tree_node, line):
    if tree_node['feat'] in line:
        if type(tree_node['left_child']) != dict:
            return tree_node['left_child']
        else:
            return make_predictions(tree_node['left_child'], line)
    else:
        if type(tree_node['right_child']) != dict:
            return tree_node['right_child']
        else:
            return make_predictions(tree_node['right_child'], line)


# *******************************************************************************************
# **********************************Start of Main********************************************
# *******************************************************************************************
f1 = open('hw1.train.col', encoding='utf8')
f2 = open('hw1.dev.col', encoding='utf8')
f3 = open('hw1.test.col', encoding='utf8')
fLines1 = f1.readlines()
fLines2 = f2.readlines()
fLines3 = f3.readlines()
fAryTrain = []
fAryDev = []
fAryTest = []

percent = 1.0
max_height = 3

fill_ary(fLines1, fAryTrain, len(fLines1)*percent)
fill_ary(fLines2, fAryDev, len(fLines2))
fill_ary(fLines3, fAryTest, len(fLines3))

features = ['or not', 'determine', 'climate', 'hot', 'sun', 'snow', 'wind', 'cold', 'temperature', 'temperatures', 'sunny', 'choose', 'but', 'and', 'forecast', 'forecaster', 'hurricane', 'tornado', 'disaster', 'severe', 'warning', '$', 'blizzard', 'white house']
prediction_train = decision_tree(fAryTrain, fAryTrain, features, max_height)
prediction_dev = decision_tree(fAryTrain, fAryDev, features, max_height)
prediction_test = decision_tree(fAryTrain, fAryTest, features, max_height)
actual_train = [line[0] for line in fAryTrain]
actual_dev = [line[0] for line in fAryDev]
actual_test = [line[0] for line in fAryTest]

percentages_train = calculate_percentage(actual_train, prediction_train)
print(f'Percentages for Training: ')
print(f'Accuracy: {percentages_train[0]}%')
print(f'Recall: {percentages_train[1]}%')
print(f'Precision: {percentages_train[2]}%\n')

percentages_dev = calculate_percentage(actual_dev, prediction_dev)
print(f'Percentages for Dev: ')
print(f'Accuracy: {percentages_dev[0]}%')
print(f'Recall: {percentages_dev[1]}%')
print(f'Precision: {percentages_dev[2]}%\n')

percentages_test = calculate_percentage(actual_test, prediction_test)
print(f'Percentages for Test: ')
print(f'Accuracy: {percentages_test[0]}%')
print(f'Recall: {percentages_test[1]}%')
print(f'Precision: {percentages_test[2]}%')
