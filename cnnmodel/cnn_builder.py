from cnnmodel import cnn_vgg, cnn_resnet, cnn_shakepyramid, cnn_resnet2

def buildNetwork(model_name = None, num_of_classes = 10, num_of_features = 512, init_weights = True, is_cifar = True):
    """ Build a CNN.

    Keyword arguments:
    model_name -- the model name
    num_of_classes -- the number of class
    num_of_features -- the number of flatten features
    init_weights -- the flag (True: initialize weights)
    is_cifar -- the flag (True: CIFAR dataset is used)
    """
    if 'VGG' in model_name:
        net_object = cnn_vgg.CnnVgg(vgg_name = model_name, num_of_classes = num_of_classes, num_of_features = num_of_features, init_weights = init_weights, is_cifar = is_cifar)
    elif 'ResNet2' in model_name:
        net_object = cnn_resnet2.CnnResNet2(resnet_name = model_name, num_of_classes = num_of_classes, num_of_features = num_of_features, init_weights = init_weights, is_cifar = is_cifar)
    elif 'ResNet' in model_name:
        net_object = cnn_resnet.CnnResNet(resnet_name = model_name, num_of_classes = num_of_classes, num_of_features = num_of_features, init_weights = init_weights, is_cifar = is_cifar)
    elif 'Shake' in model_name:
        net_object = cnn_shakepyramid.CnnShakePyramid(shakepyramid_name = model_name, num_of_classes = num_of_classes, init_weights = init_weights)
    else:
        net_object = None

    return net_object
