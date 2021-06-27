
import os
import glob

def get_sample_number(sample_dir): #eg:  '/content/dataset/test/blink/100/10/02685.bmp' -> 2685
  return int(sample_dir.split('/')[-1][:-4])

def img_lst(file_path):
  file_lst = []
  if 'unblink' in file_path: y = 1
  else: y = 0

  for file in os.listdir(file_path):
    if file.endswith('.bmp') : file_lst.append(f'{file_path}/{file}')
  return sorted(file_lst, key= get_sample_number), y

def get_dataset_filedirvslabel(data_dir):
  dataset=[]
  for data_dir_subject in glob.glob(f'{data_dir}/*'):
    if not os.path.isdir(data_dir_subject):continue
    dataset.append(img_lst(f'{data_dir_subject}/10'))
  return dataset