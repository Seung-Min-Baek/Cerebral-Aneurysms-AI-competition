'''
test값에 대해서 예측하고 결과를 output.csv파일로 저장하는 코드입니다.

test에 사용할 이미지를 불러와서 예측후 Aneurysm 에 대해 0과 1사이의 실수값,
21개 뇌부위들에 대해서 0 혹은 1의 정수값을 예측한 결과를 output.csv 파일로 저장

'''

from tensorflow import keras
import matplotlib.pyplot as plt

import cv2
import glob
import pandas as pd

import numpy as np
from tensorflow.keras.models import load_model
from tensorflow.keras import optimizers
from tensorflow.keras import models
from tensorflow.keras import layers
import matplotlib.pyplot as plt
from tensorflow.keras.utils import plot_model   
import tensorflow as tf

# 테스트 이미지 load

# 이미지를 저장할 리스트
image_list = []

# 예시 image_files = glob.glob('C:/Users/USER/Desktop/data/2023_k_ium_composition/train_set/*.jpg')
image_files = glob.glob('C:/Users/USER/Desktop/brain_aneurysm_contest/test_set_0719/test_set/*.jpg')


# 이미지 파일을 하나씩 불러와 리스트에 추가
for image_file in image_files:
    image = cv2.imread(image_file)
    # 이미지 크기를 동일하게 조정
    image = cv2.resize(image, (224, 224)) 
    #리스트에 추가
    image_list.append(image)

# 이미지 리스트를 np.array로 변환
x_train = np.array(image_list)


########################################################################

#  테스트 레이블 load

# CSV 파일 경로
# 예시 csv_file = "C:/Users/USER/Desktop/data/2023_k_ium_composition/train_set/train.csv"
csv_file = "C:/Users/USER/Desktop/brain_aneurysm_contest/test_set_0719/test_set/test.csv"


# CSV 파일 읽기
data = pd.read_csv(csv_file)

# 열 이름 리스트
column_names = ['L_ICA', 'R_ICA', 'L_PCOM', 'R_PCOM', 'L_AntChor', 'R_AntChor', 'L_ACA', 'R_ACA',
                'L_ACOM', 'R_ACOM', 'L_MCA', 'R_MCA', 'L_VA', 'R_VA', 'L_PICA', 'R_PICA', 'L_SCA',
                'R_SCA', 'BA', 'L_PCA', 'R_PCA']

# 데이터를 저장할 리스트
vector_list = []

# 각 열에 해당하는 값을 가져와서 벡터로 만들어 리스트에 추가
for column_name in column_names:
    column_values = data[column_name].values
    vector_list.append(column_values)

# 리스트의 요소를 열 방향으로 합쳐서 NumPy 배열로 변환
t_train = np.column_stack(vector_list) 


print(t_train.shape)


########################################################

# 'Index' 열의 값 가져오기
aneurysm_values = data['Index'].values

# NumPy 배열로 변환
Index_ = np.array(aneurysm_values)


####################################################################


# 미리 학습한 모델 로드
model = tf.keras.models.load_model('aug_efficientnet.h5')

t_pred = model.predict(x_train) 

t_pred_reshaped = t_pred.reshape(-1, 8, 21)

# 각 8개의 행 그룹마다 축 1을 따라 최댓값을 찾습니다.
max_values = np.amax(t_pred_reshaped, axis=1)

final_predictions = max_values.reshape(-1, 21)

# 임계값(threshold) 설정하여 값 변환
threshold = 0.98
t_pred_binary = np.where(final_predictions > threshold, 1, 0)


def transform_predictions(max_values):
    final_predictions = max_values.reshape(-1, 21)
    transformed_values = []
    for row in final_predictions:
        if np.any(row > 0.5):
            transformed_values.append(np.max(row))
        else:
            transformed_values.append(np.min(row))
    return np.array(transformed_values)

# 변환 및 반환 작업 수행
transformed_values = transform_predictions(max_values)


##############################################################

# output.csv 파일 저장

header = ['Index', 'Aneurysm', 'L_ICA', 'R_ICA', 'L_PCOM', 'R_PCOM', 'L_AntChor', 'R_AntChor', 'L_ACA', 'R_ACA', 'L_ACOM', 'R_ACOM', 'L_MCA', 'R_MCA', 'L_VA', 'R_VA', 'L_PICA', 'R_PICA', 'L_SCA', 'R_SCA', 'BA', 'L_PCA', 'R_PCA']

# 데이터 프레임 생성
df = pd.DataFrame(columns=header)

df.iloc[:,0] = Index_
df.iloc[:,1] = transformed_values
df.iloc[:,2:] = t_pred_binary

# CSV 파일로 저장
df.to_csv('output.csv', index=False)
