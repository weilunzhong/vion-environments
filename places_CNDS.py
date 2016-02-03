from kaffe.tensorflow import Network

class PlacesCNDS(Network):
    scale_size = (227,227)
    mean_vec = [103.939, 116.799, 123.68]
    def setup(self):
        (self.feed('data')
             .conv(7, 7, 64, 2, 2, name='conv1')
             .max_pool(3, 3, 2, 2, name='pool1')
             .conv(3, 3, 128, 1, 1, name='conv2')
             .max_pool(2, 2, 2, 2, name='pool2')
             .conv(3, 3, 256, 1, 1, name='conv3_1')
             .conv(3, 3, 256, 1, 1, name='conv3_2')
             .max_pool(2, 2, 2, 2, name='pool3')
             .conv(3, 3, 512, 1, 1, name='conv4_1')
             .conv(3, 3, 512, 1, 1, name='conv4_2')
             .max_pool(2, 2, 2, 2, name='pool4')
             .conv(3, 3, 512, 1, 1, name='conv5_1')
             .conv(3, 3, 512, 1, 1, name='conv5_2')
             .max_pool(2, 2, 2, 2, name='pool5')
             .fc(4096, name='fc6')
             .fc(4096, name='fc7')
             .fc(205, relu=False, name='fc8'))
