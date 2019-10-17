'''
argvs
1 : out_dir
2 : architecture
3 : gpu id
'''

import os
import sys
os.environ["CUDA_DEVICE_ORDER"]="PCI_BUS_ID"
os.environ["CUDA_VISIBLE_DEVICES"]=sys.argv[len(sys.argv)-1]
import numpy as np
import keras
import datetime

'''parse arguments'''
tag = datetime.datetime.now().strftime('%Y%m%d%H%M%S')
out_dir = tag + '_' + sys.argv[1]
archi = sys.argv[2]

image_size = (240, 320)
num_class = 2

BATCH_SIZE = 32
EPOCHS = 100195
EARLY_STOP = 5

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
    print ('invalid architecture...')
    exit()

adam = keras.optimizers.Adam(lr=0.0001)
model.compile(loss='categorical_crossentropy', optimizer=adam, metrics=['acc'])
# model.load_weights('resnext50_pipe97Kpatch_lr0.001/checkpoint.hdf5')

'''check point'''

filepath="%s/checkpoint_{val_acc:.4f}.hdf5"%(out_dir)
checkpoint = keras.callbacks.ModelCheckpoint(filepath, monitor='val_acc', 
                    verbose=1, save_best_only=True, mode='max')

tbCallback = keras.callbacks.TensorBoard(log_dir='%s/Graph'%(out_dir), histogram_freq=0,  
          write_graph=True, write_images=True)
    
early_stop_cb =  keras.callbacks.EarlyStopping(monitor='val_acc', mode='max',
          min_delta=0, patience=EARLY_STOP, verbose=1)

'''prepare out_dir'''

log_filename = '%s/log_%s.txt'%(out_dir, out_dir)
if os.path.exists(out_dir):
    raise "%s exists"%(out_dir)
else:
    os.makedirs(out_dir)

# class My_Callback(keras.callbacks.Callback):
#     def on_epoch_end(self, epoch, logs={}):
#         test_loss, test_acc = model.evaluate(x_test, y_test)

#         global log_test_loss
#         global log_test_acc
#         log_test_acc += [test_acc]
#         log_test_loss += [test_loss]

#         print 'test_loss', test_loss, 'test_acc', test_acc
#         return

# load data
from DataGenerator import DataGenerator
# Datasets
data_meta = np.load('data/meta.npy')
partition = data_meta[0]
labels = data_meta[0]['labels']

# Generators
training_generator = DataGenerator(partition['train'], labels)
validation_generator = DataGenerator(partition['val'], labels)

'''train model'''
history = model.fit_generator(generator=training_generator,
                    validation_data=validation_generator,
                    epochs=EPOCHS,
                    use_multiprocessing=True,
                    callbacks=[checkpoint, tbCallback, early_stop_cb],
                    verbose=1,
                    workers=6)

# log
log_acc = history.history['acc']
log_val_acc = history.history['val_acc']
log_loss = history.history['loss']
log_val_loss = history.history['val_loss']

with open(log_filename, 'at') as f:
        f.write('train_acc\tval_acc\ttrain_loss\tval_loss\n')
        for i in range(len(log_acc)):
            f.write('%g\t%g\t%g\t%g\n'%(log_acc[i], log_val_acc[i],
                                        log_loss[i], log_val_loss[i]))

        ind = len(log_acc) - EARLY_STOP - 1
        f.write('#################\nresults:\n')
        f.write('*     #epoch :', ind+1, '\n')
        f.write('*  train acc :', log_acc[ind], '\n')
        f.write('*    val acc :', log_val_acc[ind], '\n')
        f.write('* train loss :', log_loss[ind], '\n')
        f.write('*   val loss :', log_val_loss[ind], '\n')

ind = len(log_acc) - EARLY_STOP - 1
print('#################\nresults:\n')
print('*     #epoch :', ind+1)
print('*  train acc :', log_acc[ind])
print('*    val acc :', log_val_acc[ind])
print('* train loss :', log_loss[ind])
print('*   val loss :', log_val_loss[ind])
