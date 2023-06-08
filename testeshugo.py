import math
import scipy.io
import pandas as pd
import numpy as np
import torch
from torch.utils.data import random_split
from torch.utils.data import DataLoader

PATH = './'
PATH_TRAIN_CSV = './data/train.csv'
PATH_TRAIN_IMG = './data/train_data.mat'
PATH_TEST_CSV = './data/test.csv'
PATH_TEST_IMG = './data/test_data.mat'

def get_data_from_mat(train_file,test_file):
    train_mat = scipy.io.loadmat(train_file) 
    test_mat = scipy.io.loadmat(test_file) 
    # print(train_mat.keys())
    # print(test_mat.keys())
    train_np = np.array(train_mat['train_data']).transpose(2,0,1)
    test_np = np.array(test_mat['test_data']).transpose(2,0,1)
    # print(train_np.shape)
    # print(test_np.shape)
    return  train_np, test_np

def load_data(path_train_csv, path_train_img, path_test_csv, path_test_img):
    train_csv = pd.read_csv(path_train_csv, header=0)
    test_csv = pd.read_csv(path_test_csv, header=0)
    train_img, test_img = get_data_from_mat(path_train_img, path_test_img)
    # train = train_csv + train_img
    # test = test_csv + test_img
    # return train, test
    return train_csv, test_csv, train_img, test_img

train_csv, test_csv, train_img, test_img = load_data(PATH_TRAIN_CSV, PATH_TRAIN_IMG, PATH_TEST_CSV, PATH_TEST_IMG)
# 1 Female 0 Male
def fix_sex(csv):
    female = csv['sex']
    male = []
    for person in female:
        if person==1:
            male.append(0)
        else:
            male.append(1)
    csv.drop('sex', axis='columns', inplace=True)
    csv['Female']=female
    csv['Male']=male
    return csv
    
train_csv = fix_sex(train_csv)
test_csv = fix_sex(test_csv)

def img_to_list(img):
    tamanho = len(img)
    lista = []
    # triangular inferior sem diagonal
    for linha in range(tamanho):
        for coluna in range(linha):
            lista.append(img[linha][coluna])
    return lista

def imgs_to_matrix(imgs):
    matrix = []
    for img in imgs:
        matrix.append(img_to_list(img))
    return matrix

def remove_null_columns(matrix1, matrix2):
    columns_to_remove = []
    m1_row_len = len(matrix1)
    m2_row_len = len(matrix2)
    column_len = len(matrix1[0])
    for column in range(column_len):
        all_zero = True
        for row in range(m1_row_len):
            if matrix1[row][column]!=0:
                all_zero=False
        for row in range(m2_row_len):
            if matrix2[row][column]!=0:
                all_zero=False
        if all_zero:
            columns_to_remove.append(column)

    brain_activity_index = list(range(column_len))
    for column in columns_to_remove[::-1]:
        brain_activity_index.pop(column)
        for line in range(m1_row_len):
            matrix1[line].pop(column)
        for line in range(m2_row_len):
            matrix2[line].pop(column)
    return matrix1, matrix2, brain_activity_index

def join_data(train_csv, test_csv, train_img, test_img):
    train_matrix = imgs_to_matrix(train_img)
    test_matrix = imgs_to_matrix(test_img)
    train_clean_matrix, test_clean_matrix, brain_activity_index = remove_null_columns(train_matrix, test_matrix)
    train_data  = pd.concat([train_csv, pd.DataFrame(train_clean_matrix)], axis=1)
    train_data .columns = list(train_csv.columns) + [f'rel-{brain_activity_index[i]}' for i in range(len(brain_activity_index))]
    test_data = pd.concat([test_csv, pd.DataFrame(test_clean_matrix)], axis=1)
    test_data.columns = list(test_csv.columns) + [f'rel-{brain_activity_index[i]}' for i in range(len(brain_activity_index))]
    return train_data , test_data, brain_activity_index

train_data, test_data, brain_activity_index = join_data(train_csv, test_csv, train_img, test_img)

print(train_data)