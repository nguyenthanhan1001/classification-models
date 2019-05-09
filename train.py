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

out_dir = sys.argv[1]
archi = sys.argv[2]

image_size = 224
num_class = 2
# gpu_id = sys.argv[2]

BATCH_SIZE = 32
EPOCHS = 100195
EARLY_STOP = 10

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
#model.load_weights('')

'''check point'''

filepath="%s/checkpoint.hdf5"%(out_dir)
checkpoint = keras.callbacks.ModelCheckpoint(filepath, monitor='val_acc', 
                    verbose=1, save_best_only=True, mode='max')

tbCallback = keras.callbacks.TensorBoard(log_dir='%s/Graph'%(out_dir), histogram_freq=0,  
          write_graph=True, write_images=True)
    
early_stop_cb =  keras.callbacks.EarlyStopping(monitor='val_acc', mode='max',
          min_delta=0, patience=EARLY_STOP, verbose=1)

'''prepare out_dir'''

log_filename = '%s/log_%s.txt'%(out_dir, out_dir)
if os.path.exists(out_dir):
    os.system('rm -rf %s/*'%(out_dir))
else:
    os.makedirs(out_dir)

'''load data'''

'''train model'''

log_test_acc = []
log_test_loss = []

# class My_Callback(keras.callbacks.Callback):
#     def on_epoch_end(self, epoch, logs={}):
#         test_loss, test_acc = model.evaluate(x_test, y_test)

#         global log_test_loss
#         global log_test_acc
#         log_test_acc += [test_acc]
#         log_test_loss += [test_loss]

#         print 'test_loss', test_loss, 'test_acc', test_acc
#         return

x_train = np.load('/home/s1710453/mydata/datasets/ISIC/exp_data/HAM10K_npy/miss_classify_res50_test/miss_images/bg_set/npy_classify_v2/train_img.npy')
y_train = np.load('/home/s1710453/mydata/datasets/ISIC/exp_data/HAM10K_npy/miss_classify_res50_test/miss_images/bg_set/npy_classify_v2/train_lbl.npy')
x_val   = np.load('/home/s1710453/mydata/datasets/ISIC/exp_data/HAM10K_npy/miss_classify_res50_test/miss_images/bg_set/npy_classify_v2/val_img.npy')
y_val   = np.load('/home/s1710453/mydata/datasets/ISIC/exp_data/HAM10K_npy/miss_classify_res50_test/miss_images/bg_set/npy_classify_v2/val_lbl.npy')

history = model.fit(x=x_train,
                    y=y_train,
                    batch_size=BATCH_SIZE,
                    epochs=EPOCHS,
                    validation_data=(x_val, y_val),
                    callbacks=[checkpoint, tbCallback, early_stop_cb],
                    verbose=1)

del x_train
del y_train
del x_val
del y_val

# evaluation
x_test  = np.load('/home/s1710453/mydata/datasets/ISIC/exp_data/HAM10K_npy/miss_classify_res50_test/miss_images/bg_set/npy_classify_v2/test_img.npy')
y_test  = np.load('/home/s1710453/mydata/datasets/ISIC/exp_data/HAM10K_npy/miss_classify_res50_test/miss_images/bg_set/npy_classify_v2/test_lbl.npy')

test_loss, test_acc = model.evaluate(x_test, y_test, verbose=1)
print 'test_loss', test_loss, 'test_acc', test_acc

# eval-set
x_eval  = np.load('/home/s1710453/mydata/datasets/ISIC/exp_data/HAM10K_npy/miss_classify_res50_test/miss_images/bg_set/npy_classify_v2/eval_set_img.npy')
y_eval  = np.load('/home/s1710453/mydata/datasets/ISIC/exp_data/HAM10K_npy/miss_classify_res50_test/miss_images/bg_set/npy_classify_v2/eval_set_lbl.npy')

eval_loss, eval_acc = model.evaluate(x_eval, y_eval, verbose=1)
print 'eval_loss', eval_loss, 'eval_acc', eval_acc

# log
log_acc = history.history['acc']
log_val_acc = history.history['val_acc']
log_loss = history.history['loss']
log_val_loss = history.history['val_loss']

log_test_acc = [0.] * len(log_acc)
log_test_loss = [0.] * len(log_acc)

with open(log_filename, 'at') as f:
        f.write('train_acc\tval_acc\ttest_acc\ttrain_loss\tval_loss\ttest_loss\n')
        for i in range(len(log_acc)):
            f.write('%g\t%g\t%g\t%g\t%g\t%g\n'%(log_acc[i], log_val_acc[i], log_test_acc[i],
                                                log_loss[i], log_val_loss[i], log_test_loss[i]))

        f.write('acc        : %g at epoch: %d\n'%(max(log_acc), 
                                        log_acc.index(max(log_acc))))
        f.write('val_acc    : %g at epoch: %d\n'%(max(log_val_acc),
                                        log_val_acc.index(max(log_val_acc))))
        #f.write('test_acc   : %g at epoch: %d\n'%(max(log_test_acc),
        #                                log_test_acc.index(max(log_test_acc))))
        f.write('test_acc   : %g\n'%(test_acc))
        f.write('loss       : %g at epoch: %d\n'%(min(log_loss),
                                        log_loss.index(min(log_loss))))
        f.write('val_loss   : %g at epoch: %d\n'%(min(log_val_loss),
                                        log_val_loss.index(min(log_val_loss))))
        #f.write('test_loss  : %g at epoch: %d\n'%(min(log_test_loss),
        #                                log_test_loss.index(min(log_test_loss))))
        f.write('test_loss  : %g\n'%(test_loss))
        f.write('#epochs    : %d\n'%(len(log_acc)))
        
        ind = len(log_acc) - EARLY_STOP - 1
        f.write('results:\n #epochs: %d\n'%(ind+1))
        f.write('%g\t%g\t%g\t%g\t%g\t%g\n'%(log_acc[ind], log_val_acc[ind], test_acc,
                                                log_loss[ind], log_val_loss[ind], test_loss))

        f.write('eval_loss: %g, eval_acc: %g\n'%(eval_loss, eval_acc))

print('acc          : %g at epoch: %d\n'%(max(log_acc), 
                                        log_acc.index(max(log_acc))))
print('val_acc      : %g at epoch: %d\n'%(max(log_val_acc),
                                log_val_acc.index(max(log_val_acc))))
print('test_acc     : %g\n'%(test_acc))
print('loss         : %g at epoch: %d\n'%(min(log_loss),
                                log_loss.index(min(log_loss))))
print('val_loss     : %g at epoch: %d\n'%(min(log_val_loss),
                                log_val_loss.index(min(log_val_loss))))
print('test_loss    : %g\n'%(test_loss))
print('#epochs      : %d\n'%(len(log_acc)))

ind = len(log_acc) - EARLY_STOP - 1
print('results:\n %s\n#epochs: %d\n'%(archi ,ind+1))
print('%g\t%g\t%g\t%g\t%g\t%g\n'%(log_acc[ind], log_val_acc[ind], test_acc,
                                                log_loss[ind], log_val_loss[ind], test_loss))

print('eval_loss: %g, eval_acc: %g\n'%(eval_loss, eval_acc))