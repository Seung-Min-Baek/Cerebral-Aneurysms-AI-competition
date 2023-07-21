'''
데이터 증식 , 모델 훈련, 저장 하는 코드입니다

부족한 훈련데이터를 보충하기 위해서 augmentation을 활용하여 데이터 양을 늘려 efficientnet 구조를 활용하여 모델을 작성하고
aug_efficientnet.h5 로 모델을 저장하였습니다

모델의 구조도는 aug_efficientnet.png 파일을 통해 확인할수 있습니다.

모델 훈련시에 사용되는 파일의 경로 잘 지정하여 사용해주세요.



#라이브러리 버전
imgaug version: 0.4.0
tensorflow version: 2.13.0
cv2 version: 4.7.0
pandas version: 1.4.4
numpy version: 1.23.5

'''

import imgaug.augmenters as iaa
from tensorflow import keras

import cv2
import glob
import pandas as pd
import numpy as np
from tensorflow.keras.utils import plot_model   

import tensorflow as tf
from tensorflow.keras.applications import EfficientNetB0
from tensorflow.keras.layers import GlobalAveragePooling2D, Dense
from tensorflow.keras.models import Sequential
from tensorflow.keras.callbacks import ModelCheckpoint   
import imgaug.augmenters as iaa
import numpy as np

# Augmentation을 위한 객체 생성
augmenter_flip = iaa.Fliplr(1.0)                 # 좌우 반전
augmenter_rotate = iaa.Affine(rotate=(-180, 180)) # 회전
augmenter_translate = iaa.Affine(translate_percent={"x": (-0.2, 0.2), "y": (-0.2, 0.2)})  # 이동
augmenter_shear = iaa.Affine(shear=(-45, 45))     #  전단
augmenter_brightness = iaa.Multiply((0.8, 1.2))   #  밝기 조절

# 이미지를 저장할 리스트
image_list = []

# 모델 훈련을 위한 train 이미지들 load
# 예시 image_files = glob.glob('C:/Users/USER/Desktop/brain_aneurysm_contest/2023_k_ium_composition/train_set/*.jpg')
image_files = glob.glob('훈련에 사용되는 이미지 폴더 경로/*.jpg')

# 원본 이미지 생성 및 augmentation 적용
for image_file in image_files:
    image = cv2.imread(image_file)
    # 이미지 크기를 동일하게 조정
    image = cv2.resize(image, (224, 224))
    
    # 원본 이미지 추가
    image_list.append(image)
    
    # 좌우 반전 augmentation
    augmented_flip = augmenter_flip.augment_image(image)
    image_list.append(augmented_flip)
    
    # 회전 augmentation
    augmented_rotate = augmenter_rotate.augment_image(image)
    image_list.append(augmented_rotate)
    
    # 이동 augmentation
    augmented_translate = augmenter_translate.augment_image(image)
    image_list.append(augmented_translate)
    
    # 전단 augmentation
    augmented_shear = augmenter_shear.augment_image(image)
    image_list.append(augmented_shear)
    
    # 밝기 조절 augmentation
    augmented_brightness = augmenter_brightness.augment_image(image)
    image_list.append(augmented_brightness)

# 이미지 리스트를 np.array로 변환
x_train = np.array(image_list)

#######################################################################

# CSV 파일 경로
# 예시 csv_file = "C:/Users/USER/Desktop/brain_aneurysm_contest/2023_k_ium_composition/train_set/train.csv"
csv_file = " 훈련에 사용할 csv 파일의 경로 "

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

# 8배(1사람당 8장) * 6배(augmentation) 
t_train = np.repeat(t_train, 48,axis=0)   

print(t_train.shape)

###########################################################################

# augmentation 한 데이터를 이용해 모델의 훈련,검증,테스트에 활용하도록 나누었습니다.

x_test = x_train[52800:,:]         # 1296
x_val = x_train[44000:52800,:]     # 8800
x_train = x_train[:44000,:]        # 44000


t_test = t_train[52800:,:]         # 1296
t_val = t_train[44000:52800,:]     # 8800
t_train = t_train[:44000,:]        # 44000

#############################################################################

# 모델 구축
model = Sequential()
model.add(EfficientNetB0(include_top=False, input_shape=(224, 224, 3)))  # EfficientNetB0의 합성곱 기반부분만 가져옵니다.
model.add(GlobalAveragePooling2D())
model.add(Dense(21, activation='sigmoid'))  # 21개의 뇌 부위에 대한 이진 분류를 위해 sigmoid 활성화 함수를 사용합니다.

# 모델 컴파일
model.compile(optimizer='adam',
              loss='binary_crossentropy',
              metrics=['accuracy'])

# 모델 훈련 중 가장 좋은 성능을 보인 모델 저장을 위한 콜백
checkpoint = ModelCheckpoint('aug_efficientnet.h5', monitor='val_accuracy', save_best_only=True, mode='max', verbose=1, save_format='h5')

# 모델 훈련
history = model.fit(x_train, t_train, epochs=10, batch_size=32, validation_data=(x_val, t_val), callbacks=[checkpoint])

# 모델 구조 시각화
plot_model(model, to_file='aug_efficientnet.png', show_shapes=True)
