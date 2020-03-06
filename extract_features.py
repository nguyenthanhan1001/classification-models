import cv2
import os
import sys
# os.environ["CUDA_DEVICE_ORDER"]="PCI_BUS_ID"
# os.environ["CUDA_VISIBLE_DEVICES"]=sys.argv[len(sys.argv)-1]
import numpy as np
import keras
import glob

'''parse arguments'''

in_dir = sys.argv[1]
out_dir = sys.argv[2]
archi = 'ResNet152'
image_size = 224
num_class = 1000
#weigths_path = sys.argv[3]
BATCH_SIZE = 32

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
    print('invalid architecture...')
    exit()

adam = keras.optimizers.Adam(lr=0.0001)
model.compile(loss='categorical_crossentropy', optimizer=adam, metrics=['acc'])
# model.load_weights(weigths_path)

''' evaluate '''

# load data
files = glob.glob(in_dir+'/*/*.jpg')
files.sort()
print('#files:', len(files))
extracted_files = [os.path.basename(it)[:-4] for it in glob.glob(out_dir+'/*/*.npy')]
print('#extracted files:', len(extracted_files))
files = [it for it in files if os.path.basename(it) not in extracted_files]
print('#files:', len(files))

# extract
def load_batch(files):
    batch = np.zeros([len(files), image_size, image_size, 3])
    for i, ff in enumerate(files):
        try:
            img = cv2.imread(ff)
            img = cv2.resize(img, (224, 224))
            batch[i] = np.array(img, dtype=np.float) / 255.
        except:
            print(ff)
    return batch, len(batch)
    pass

for i in range(0, len(files), BATCH_SIZE):
    batch, count = load_batch(files[i:i+BATCH_SIZE])
    outs = model.predict(batch)

    for k in range(count):
        out_fd = os.path.dirname(files[i+k])
        out_fd = out_fd.replace(os.path.dirname(out_fd), out_dir)
        if not os.path.exists(out_fd):
            os.makedirs(out_fd)
        np.save(out_fd+'/'+os.path.basename(files[i+k])+'.npy', outs[k])
        pass
    print('%d/%d'%(i+BATCH_SIZE, len(files)), end='\r')

# corrupted files
# ../../Volumes/Samsung_T5/DATASETS/LSC2020/2016-08-18/20160818_142810_000.jpg
# ../../Volumes/Samsung_T5/DATASETS/LSC2020/2016-08-31/20160831_083535_000.jpg
# ../../Volumes/Samsung_T5/DATASETS/LSC2020/2016-09-01/20160901_120054_000.jpg
# ../../Volumes/Samsung_T5/DATASETS/LSC2020/2016-09-12/20160912_122440_000.jpg
