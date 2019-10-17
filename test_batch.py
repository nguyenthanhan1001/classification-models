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

out_dir = sys.argv[1]
archi = sys.argv[2]
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

# load data
from DataGenerator import DataGenerator

# Datasets
data_meta = np.load('data/meta.npy')
partition = data_meta[0]
labels = data_meta[0]['labels']

# Generators
# training_generator = DataGenerator(partition['train'], labels,)
# validation_generator = DataGenerator(partition['val'], labels)
test_generator = DataGenerator(partition['test'], labels,)

#print 'train:', model.evaluate_generator(training_generator, workers=6, use_multiprocessing=True, verbose=1)
# print 'val:', model.evaluate_generator(validation_generator, workers=6, use_multiprocessing=True, verbose=1)
# print 'test:', model.evaluate_generator(test_generator, workers=6, use_multiprocessing=True, verbose=1)

lbl = []
for i, ID in enumerate(test_generator.list_IDs):
    vid = ID.split('/')[-1].split('_')[0]
    lbl.append(test_generator.labels[vid])

np.save(out_dir + '/labels_test', lbl)

pre = model.predict_generator(test_generator, workers=6, use_multiprocessing=True, verbose=1)
np.save(out_dir + '/predictions_test', pre)