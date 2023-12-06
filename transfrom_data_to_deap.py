import glob
import os
import pickle
import numpy as np
import pandas as pd


def get_filenames_in_folder(folder_path):
    filenames = glob.glob(os.path.join(folder_path, '*'))
    filenames = [os.path.basename(filename) for filename in filenames]
    filenames.sort(key=lambda x: int(x.split('-')[0]))
    return filenames


# 指定需要修改格式的文件目录和保存目录
folder_path = './data/'
save_path = './deap_format'

# 获取文件名数组
files = get_filenames_in_folder(folder_path)
# 提取每个文件名中的组号
group_numbers = set([filename.split('-')[0] for filename in files])
# 计算不同组的数量
number_of_groups = len(group_numbers)
print(f"总共有 {number_of_groups} 组")

hvha_data = []
hvla_data = []
lvha_data = []
lvla_data = []

for i in range(len(files)):
    # 读取.dat文件
    trial = pd.read_csv(folder_path + files[i], delimiter='\t')
    if files[i].split('.')[0].split('-')[1] == 'hvha':
        hvha_data.append(trial)
    elif files[i].split('.')[0].split('-')[1] == 'hvla':
        hvla_data.append(trial)
    elif files[i].split('.')[0].split('-')[1] == 'lvha':
        lvha_data.append(trial)
    elif files[i].split('.')[0].split('-')[1] == 'lvla':
        lvla_data.append(trial)

######################################
# temp = lvla_data[0]
# points_per_channel = temp['# data'].apply(lambda x: len(x.split(',')))
# # 打印每个通道的数据点数
# for i, points in enumerate(points_per_channel):
#     print(f"通道 {i + 1} 有 {points} 个数据点")
#
# temp = lvla_data[1]
# points_per_channel = temp['# data'].apply(lambda x: len(x.split(',')))
# # 打印每个通道的数据点数
# for i, points in enumerate(points_per_channel):
#     print(f"通道 {i + 1} 有 {points} 个数据点")

# # 假设 lvla_data[0] 是一个 DataFrame，并且 '# data' 列包含了字符串形式的数据点
# temp = lvla_data[1]
# # 将字符串转换为浮点数列表
# temp['# data'] = temp['# data'].apply(lambda x: [float(num) for num in x.split(',')])
# # 计算所有通道中最大的数据点数量
# max_points = temp['# data'].apply(len).max()
# # 对于每个通道，将数据点扩展到最大长度
# temp['# data'] = temp['# data'].apply(lambda channel_data: channel_data + [np.nan] * (max_points - len(channel_data)))
# # 创建一个 (32, x) 的 NumPy 数组，其中 x 是最大的数据点数量
# data_array = np.array(temp['# data'].tolist())

# 标签数组
# high = np.repeat([[1]], 830016, axis=0)
# low = np.repeat([[0]], 830016, axis=0)
# lvha_label = np.concatenate((low, high), axis=1)
# lvha_label = np.reshape(lvha_label, (2, 32, 25938))
# lvla_label = np.concatenate((low, low), axis=1)
# lvla_label = np.reshape(lvla_label, (2, 32, 25938))
# hvla_label = np.concatenate((high, low), axis=1)
# hvla_label = np.reshape(hvla_label, (2, 32, 25938))
# hvha_label = np.concatenate((high, high), axis=1)
# hvha_label = np.reshape(hvha_label, (2, 32, 25938))
# label_array = np.concatenate((hvha_label, hvla_label, lvha_label, lvla_label), axis=2)

label_array = np.array([[1, 1], [1, 0], [0, 1], [0, 0]])

for i in range(number_of_groups):
    group_number = files[i * 4].split('-')[0]

    # 读取hvha数据
    hvha = hvha_data[i]
    # 将字符串转换为浮点数列表
    hvha['# data'] = hvha['# data'].apply(lambda x: [float(num) for num in x.split(',')])
    # 计算所有通道中最大的数据点数量
    max_points = hvha['# data'].apply(len).max()
    # 对于每个通道，将数据点扩展到最大长度
    hvha['# data'] = hvha['# data'].apply(
        lambda channel_data: channel_data + [np.nan] * (max_points - len(channel_data)))
    # 创建一个 (32, x) 的 NumPy 数组，其中 x 是最大的数据点数量
    hvha_array = np.array(hvha['# data'].tolist())

    # 读取hvla数据
    hvla = hvla_data[i]
    # 将字符串转换为浮点数列表
    hvla['# data'] = hvla['# data'].apply(lambda x: [float(num) for num in x.split(',')])
    # 计算所有通道中最大的数据点数量
    max_points = hvla['# data'].apply(len).max()
    # 对于每个通道，将数据点扩展到最大长度
    hvla['# data'] = hvla['# data'].apply(
        lambda channel_data: channel_data + [np.nan] * (max_points - len(channel_data)))
    # 创建一个 (32, x) 的 NumPy 数组，其中 x 是最大的数据点数量
    hvla_array = np.array(hvla['# data'].tolist())

    # 读取lvha数据
    lvha = lvha_data[i]
    # 将字符串转换为浮点数列表
    lvha['# data'] = lvha['# data'].apply(lambda x: [float(num) for num in x.split(',')])
    # 计算所有通道中最大的数据点数量
    max_points = lvha['# data'].apply(len).max()
    # 对于每个通道，将数据点扩展到最大长度
    lvha['# data'] = lvha['# data'].apply(
        lambda channel_data: channel_data + [np.nan] * (max_points - len(channel_data)))
    # 创建一个 (32, x) 的 NumPy 数组，其中 x 是最大的数据点数量
    lvha_array = np.array(lvha['# data'].tolist())

    # 读取lvla数据
    lvla = lvla_data[i]
    # 将字符串转换为浮点数列表
    lvla['# data'] = lvla['# data'].apply(lambda x: [float(num) for num in x.split(',')])
    # 计算所有通道中最大的数据点数量
    max_points = lvla['# data'].apply(len).max()
    # 对于每个通道，将数据点扩展到最大长度
    lvla['# data'] = lvla['# data'].apply(
        lambda channel_data: channel_data + [np.nan] * (max_points - len(channel_data)))
    # 创建一个 (32, x) 的 NumPy 数组，其中 x 是最大的数据点数量
    lvla_array = np.array(lvla['# data'].tolist())

    # 裁剪数组到统一的大小 (32, 25938)
    hvha_array_sliced = hvha_array[:, :25938]
    hvla_array_sliced = hvla_array[:, :25938]
    lvha_array_sliced = lvha_array[:, :25938]
    lvla_array_sliced = lvla_array[:, :25938]
    # 合并四个数组为一个 (4, 32, 25938) 的数组
    data_array = np.stack((hvha_array_sliced, hvla_array_sliced, lvha_array_sliced, lvla_array_sliced))
    # # 合并四个数组为一个 (32, 25938*4) 的数组
    # data_array = np.concatenate((hvha_array_sliced, hvla_array_sliced, lvha_array_sliced, lvla_array_sliced), axis=1)
    print(data_array.shape)
    print(label_array.shape)

    data_deap_format = {
        'labels': label_array,
        'data': data_array,  # 假设 data_array 是前面步骤中创建的 (32, x) 数组
    }

    # 构造新的文件名
    new_file_name = f"sample_{group_number}.dat"
    print(f"正在保存 {new_file_name} ...")
    temp_save_path = os.path.join(save_path, new_file_name)
    # 保存数据到 .dat 文件
    with open(temp_save_path, 'wb') as file:
        pickle.dump(data_deap_format, file)
