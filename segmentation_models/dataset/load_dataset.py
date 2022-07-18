import inspect
import zipfile

import gdown
import os
from google_drive_downloader import GoogleDriveDownloader as gdd
import os
import wget

from googleapiclient.http import MediaFileUpload, MediaIoBaseDownload
import io
import re

# all files end with "?usp=sharing" or "/view?usp=sharing"
DRIVE_LINK_FOLDER = 'https://drive.google.com/drive/folders/1u3To4nXF1fLoQD8cNtGHVLNZmK2pa4_f'
TRAIN_ID = '1C6IaJ1AVDFOtgWChm4-mNSC7ue8NDVTu'
TEST_ID = '1C5LuAjr2G1us-zXye47DzBUR3_0KPjUJ'
VALIDATE_10_ID = '13DHiDsINxtVW2uq0i-mR3MktzimiRMp4'
VALIDATE_100_ID = '1Yq5L78lmW_6cSfOnL7h3Ie6jYowKRC8L'
#https://drive.google.com/file/d/1C5LuAjr2G1us-zXye47DzBUR3_0KPjUJ/view?usp=sharing
IDS = [TRAIN_ID, TEST_ID, VALIDATE_10_ID, VALIDATE_100_ID]
NAMES = ['train', 'test', 'validate_10', 'validate_100']
PATH = os.path.join(os.path.dirname(os.path.abspath(inspect.stack()[0][1])), 'data')


def download_file(file_index, file_name):
    path = os.path.join(PATH, file_name + '.zip')
    #gdd.download_file_from_google_drive(file_id=IDS[file_index], dest_path=path, unzip=True)
    wget.download("https://drive.google.com/file/d/1C5LuAjr2G1us-zXye47DzBUR3_0KPjUJ", out=path)
    #unzip_file(path)



def unzip_file(path_to_zip_file):
    with zipfile.ZipFile(path_to_zip_file, 'r') as zip_ref:
        zip_ref.extractall(PATH)


def download_all_data():
    for i in range(len(IDS)):
        download_file(i, NAMES[i])


def create_dataset():
    train_dataset, test_dataset = [], []
    return train_dataset, test_dataset
