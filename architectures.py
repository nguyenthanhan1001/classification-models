import keras
from keras import models
from keras import layers
from keras import applications

def create_ResNet50(image_size, num_class):
    resnet_conv = applications.ResNet50(weights='imagenet', include_top=False,
                input_shape=(image_size, image_size, 3))

    model = models.Sequential()
    model.add(resnet_conv)
    model.add(layers.GlobalAveragePooling2D())
    model.add(layers.Dense(num_class, activation='softmax'))
    return model
    pass

def create_ResNet101(image_size, num_class):
    resnet_conv = applications.ResNet101(weights='imagenet', include_top=False,
                input_shape=(image_size, image_size, 3))

    model = models.Sequential()
    model.add(resnet_conv)
    model.add(layers.GlobalAveragePooling2D())
    model.add(layers.Dense(num_class, activation='softmax'))
    return model
    pass

def create_ResNet152(image_size, num_class):
    resnet_conv = applications.ResNet152(weights='imagenet', include_top=False,
                input_shape=(image_size, image_size, 3))

    model = models.Sequential()
    model.add(resnet_conv)
    model.add(layers.GlobalAveragePooling2D())
    model.add(layers.Dense(num_class, activation='softmax'))
    return model
    pass

def create_ResNeXt50(image_size, num_class):
    resnet_conv = applications.ResNeXt50(weights='imagenet', include_top=False,
                input_shape=(image_size, image_size, 3))

    model = models.Sequential()
    model.add(resnet_conv)
    model.add(layers.GlobalAveragePooling2D())
    model.add(layers.Dense(num_class, activation='softmax'))
    return model
    pass

def create_ResNeXt101(image_size, num_class):
    resnet_conv = applications.ResNeXt101(weights='imagenet', include_top=False,
                input_shape=(image_size, image_size, 3))

    model = models.Sequential()
    model.add(resnet_conv)
    model.add(layers.GlobalAveragePooling2D())
    model.add(layers.Dense(num_class, activation='softmax'))
    return model
    pass

def create_InceptionV3(image_size, num_class):
    resnet_conv = applications.InceptionV3(weights='imagenet', include_top=False,
                input_shape=(image_size, image_size, 3))

    model = models.Sequential()
    model.add(resnet_conv)
    model.add(layers.GlobalAveragePooling2D())
    model.add(layers.Dense(num_class, activation='softmax'))
    return model
    pass

def create_InceptionResNetV2(image_size, num_class):
    resnet_conv = applications.InceptionResNetV2(weights='imagenet', include_top=False,
                input_shape=(image_size, image_size, 3))

    model = models.Sequential()
    model.add(resnet_conv)
    model.add(layers.GlobalAveragePooling2D())
    model.add(layers.Dense(num_class, activation='softmax'))
    return model
    pass

def create_DenseNet121(image_size, num_class):
    resnet_conv = applications.DenseNet121(weights='imagenet', include_top=False,
                input_shape=(image_size, image_size, 3))

    model = models.Sequential()
    model.add(resnet_conv)
    model.add(layers.GlobalAveragePooling2D())
    model.add(layers.Dense(num_class, activation='softmax'))
    return model
    pass

def create_DenseNet169(image_size, num_class):
    resnet_conv = applications.DenseNet169(weights='imagenet', include_top=False,
                input_shape=(image_size, image_size, 3))

    model = models.Sequential()
    model.add(resnet_conv)
    model.add(layers.GlobalAveragePooling2D())
    model.add(layers.Dense(num_class, activation='softmax'))
    return model
    pass

def create_DenseNet201(image_size, num_class):
    resnet_conv = applications.DenseNet201(weights='imagenet', include_top=False,
                input_shape=(image_size, image_size, 3))

    model = models.Sequential()
    model.add(resnet_conv)
    model.add(layers.GlobalAveragePooling2D())
    model.add(layers.Dense(num_class, activation='softmax'))
    return model
    pass

def create_VGG16(image_size, num_class):
    resnet_conv = applications.VGG16(weights='imagenet', include_top=False,
                input_shape=(image_size, image_size, 3))

    model = models.Sequential()
    model.add(resnet_conv)
    model.add(layers.GlobalAveragePooling2D())
    model.add(layers.Dense(num_class, activation='softmax'))
    return model
    pass

def create_VGG19(image_size, num_class):
    resnet_conv = applications.VGG19(weights='imagenet', include_top=False,
                input_shape=(image_size, image_size, 3))

    model = models.Sequential()
    model.add(resnet_conv)
    model.add(layers.GlobalAveragePooling2D())
    model.add(layers.Dense(num_class, activation='softmax'))
    return model
    pass

def create_Xception(image_size, num_class):
    resnet_conv = applications.Xception(weights='imagenet', include_top=False,
                input_shape=(image_size, image_size, 3))

    model = models.Sequential()
    model.add(resnet_conv)
    model.add(layers.GlobalAveragePooling2D())
    model.add(layers.Dense(num_class, activation='softmax'))
    return model
    pass