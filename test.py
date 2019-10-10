'''
argvs
1 : out_dir
2 : architecture
3 : weights path
4 : gpu id
'''

import os
import sys
os.environ["CUDA_DEVICE_ORDER"]="PCI_BUS_ID"
os.environ["CUDA_VISIBLE_DEVICES"]=sys.argv[len(sys.argv)-1]
import numpy as np
import keras

'''parse arguments'''

archi = sys.argv[1]
image_size = (240, 320)
num_class = 2
weigths_path = sys.argv[3]

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
model.load_weights(weigths_path)

''' evaluate '''

x_train = np.load('data_npy_5_1_4/train_images.npy')
y_train = np.load('data_npy_5_1_4/train_labels.npy')
print model.evaluate(x_train, y_train)
del x_train
del y_train

x_val   = np.load('data_npy_5_1_4/val_images.npy')
y_val   = np.load('data_npy_5_1_4/val_labels.npy')
print model.evaluate(x_val, y_val)

del x_val
del y_val

x_test  = np.load('data_npy_5_1_4/test_images.npy')
y_test  = np.load('data_npy_5_1_4/test_labels.npy')
print model.evaluate(x_test, y_test)
del x_test
del y_test


pre = model.predict(x_test)
np.save(out_dir + '/predictions_test', pre)
np.save(out_dir + '/labels_test', y_test)