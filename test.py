'''
argvs
1 : model name
2 : architecture
3 : image size
4 : gpu id
'''

import os
import sys
# os.environ["CUDA_DEVICE_ORDER"]="PCI_BUS_ID"
# os.environ["CUDA_VISIBLE_DEVICES"]=sys.argv[len(sys.argv)-1]
import numpy as np
import keras

'''parse arguments'''

archi = sys.argv[1]

image_size = 224
num_class = 2
# gpu_id = sys.argv[2]

'''create model'''

import architectures

if archi == 'ResNet50':
    model = architectures.create_ResNet50(image_size, num_class)
elif archi == 'ResNet101':
    model = architectures.create_ResNet101(image_size, num_class)
elif archi == 'ResNet152':
    model = architectures.create_ResNet152(image_size, num_class)
elif archi == 'ResNeXt50':
    model = architectures.create_ResNeXt50(image_size, num_class)
elif archi == 'ResNeXt101':
    model = architectures.create_ResNeXt101(image_size, num_class)
elif archi == 'InceptionV3':
    model = architectures.create_InceptionV3(image_size, num_class)
elif archi == 'InceptionResNetV2':
    model = architectures.create_InceptionResNetV2(image_size, num_class)
elif archi == 'DenseNet121':
    model = architectures.create_DenseNet121(image_size, num_class)
elif archi == 'DenseNet169':
    model = architectures.create_DenseNet169(image_size, num_class)
elif archi == 'DenseNet201':
    model = architectures.create_DenseNet201(image_size, num_class)
elif archi == 'VGG16':
    model = architectures.create_VGG16(image_size, num_class)
elif archi == 'VGG19':
    model = architectures.create_VGG19(image_size, num_class)
elif archi == 'Xception':
    model = architectures.create_Xception(image_size, num_class)
else:
    print 'invalid architecture...'
    exit()

adam = keras.optimizers.Adam(lr=0.0001)
model.compile(loss='categorical_crossentropy', optimizer=adam, metrics=['acc'])
model.load_weights('pretrained/HAM10K/resnet50/checkpoint.hdf5')

# x_train = np.load('/home/s1710453/mydata/datasets/ISIC/exp_data/HAM10K_npy/train_img.npy')
# y_train = np.load('/home/s1710453/mydata/datasets/ISIC/exp_data/HAM10K_npy/train_lbl.npy')
# x_val   = np.load('/home/s1710453/mydata/datasets/ISIC/exp_data/HAM10K_npy/val_img.npy')
# y_val   = np.load('/home/s1710453/mydata/datasets/ISIC/exp_data/HAM10K_npy/val_lbl.npy')
x_test  = np.load('/home/s1710453/mydata/datasets/ISIC/exp_data/HAM10K_npy/test_img.npy')
y_test  = np.load('/home/s1710453/mydata/datasets/ISIC/exp_data/HAM10K_npy/test_lbl.npy')

pre_y = model.predict_classes(x_test)
np.save('pre_y_res50', pre_y)