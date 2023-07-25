# Cerebral-Aneurysms-AI-competition

## data
본 대회에서 제공한 데이터는 [train_set] 디렉토리 내에 9016장의 뇌동맥류를 진단할 수 있는 뇌 혈관 영상과 train.csv 파일이 존재한다. 
9016장의 영상은 총 1127명의 환자에 대하여 조영제를 주입한 위치와 촬영한 각도에 따라 각 환자당 8장의 영상이다. 
train.csv 파일에는 1127명의 환자에 대하여 뇌동맥류 여부와 뇌동맥류가 발견된 위치에 대한 값이 0과 1로 입력되어 있다. 
값이 0일때는 뇌동맥류가 발견되지 않았음을 의미하고, 값이 1일때는 뇌동맥류가 발견되었음을 의미한다. 
또한, 뇌동맥류가 발견된 환자에 대하여 해당 뇌동맥류가 발견된 위치에 1로 입력이 되어있다.

## data preprocessing
본 대회에서 제공한 데이터 총 9016의 영상에 대해서 좌우 반전, 회전, 이동, 전단, 밝기 조절을 통해서 데이터의 양을 증식시켜 훈련에 사용하였다.
총 54096개의 영상 중 44000개를 training 이미지로 사용하였고, 8800개의 영상을 validation 이미지로 사용하였다. 
나머지 1296개의 영상은 test 이미지로 사용하였다.

## model
모델은 EfficientnetB0의 구조에 GlobalAveragePooling2D 층을 추가한 뒤, 21개의 output 에 sigmoid activation을 사용하였다.
모델 compile 시 optimizer는 Adam, loss는 binary_crossentropy, metrics는 accuracy를 사용하여 훈련하였다. 
또한, 훈련 중 validation accuracy가 가장 높았을 때를 model로 ‘aug_efficientnet.h5’로 저장하였다.

## model predict
모델 학습 과정에서 저장하였던 ‘aug_efficientnet.h5’파일을 불러와서 검증하고자 하는 데이터를 입력시켜 예측값을 출력한다. 
출력된 환자 한명당 8장의 예측값을 Late fusion 기술 중 max 기법을 사용하여 8장 중 확률값이 가장 높은 예측값을 저장한다. 
그리고 임계값을 설정하여 0과 1 사이의 실수값인 예측값을 0 혹은 1의 정수값으로 21개의 뇌 부위에 대하여 정수값을 저장한다.
앞서 8장을 바탕으로 예측하였던 결과값의 각 row당 임계값 이상의 값이 있다면, row의 최대값을 저장하고, 임계값 미만의 값이 있다면, 
row의 최소값을 저장하도록 하여 Aneurysm에 실수값을 0과 1 사이의 값으로 저장하였다.
