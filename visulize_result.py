from vgg16_places2 import VggPlaces2
import tensorflow as tf
import numpy as np
from vionaux.rnd import vidioids
import cv2

class EnvronmentClassifier(object):

    def load_image_mean(self, path):
        mean = np.load(path)
        mean = mean.transpose(1,2,0)
        return mean

    def network_deployment(self, model,batch_generator, batch_size, image_size, mean):
        test_data = tf.placeholder(tf.float32, shape=([batch_size]+list(VggPlaces2.scale_size)+[3]))
        net = VggPlaces2({'data':test_data})

        with tf.Session() as sesh:
            net.load(model, sesh)
            try:
                while True:
                    batch, timestamp = next(batch_generator)
                    print batch.shape
                    assert batch.shape[3] == 3
#                    for idx in range(0, len(batch)):
#                        batch[idx] = np.subtract(batch[idx], mean)
                    frame_in_batch = batch.shape[0]
                    if frame_in_batch != batch_size:
                        temp_batch = np.ndarray([batch_size]+list(image_size)+[3])
                        temp_batch[0:frame_in_batch] = batch
                        batch = temp_batch
                        output = sesh.run(net.get_output(), feed_dict= {test_data:batch})
                        yield output[0:frame_in_batch], timestamp
                    else:
                        output = sesh.run(net.get_output(), feed_dict= {test_data:batch})
                        yield output, timestamp
            except StopIteration:
                    return
def main():
    video_path = "/mnt/movies03/boxer_movies/tt3247714/Survivor (2015)/Survivor.2015.720p.BluRay.x264.YIFY.mp4"
    cap = cv2.VideoCapture(video_path)

    VHH = vidioids.VionVideoHandler()
    batch_size = 20
    image_size = VggPlaces2.scale_size
    batch_generator = VHH.get_batches(video_path, 0.01, 1000, 2000, batch_size, image_size)
    EC = EnvronmentClassifier()
    mean = EC.load_image_mean("places205_mean.npy")
    out = EC.network_deployment('vgg16_places2_caffemodel.npy', batch_generator, batch_size, image_size, mean)
    for i, timestamp in out:
        assert i.shape[0] == len(timestamp)
        print i.argmax(axis=1)
        print timestamp
        print "one batch"
        real_time = [i/1000. for i in timestamp]
        for i in real_time:
            cap.set(cv2.cv.CV_CAP_PROP_POS_MSEC, i)
            _, frame = cap.read()
            cv2.imshow("frame", frame)
            if cv2.waitKey(-1) & 0xFF == ord('q'):
                break


if __name__ == "__main__":
    main()
