import tensorflow as tf
import numpy as np
import skimage.transform
from skimage.io import imread

with open('PlacesCNDS.tfmodel', 'rb') as f:
    fileContent = f.read()

img = imread('/home/vionlabs/Downloads/test.jpg')
img = skimage.transform.resize(img, (227,550))
img = img.reshape(1, 1, 227, 550, 3)
print img.shape
image_batch = np.concatenate((list(img)*20), axis=0)
print image_batch.shape

graph_def = tf.GraphDef()
graph_def.ParseFromString(fileContent)
input_data = tf.placeholder(tf.float32, [1,227,550,3])
tf.import_graph_def(graph_def)

sess = tf.InteractiveSession()
names = sess.graph.__dict__['_names_in_use'].keys()
valid_names = [x for x in names if x[:7]=='import/']
for x in valid_names:
    print x
inference = sess.graph.get_tensor_by_name("prob:0")
prediction = sess.run(inference, feed_dict={'Placeholder': image_batch})
