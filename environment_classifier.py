from places_CNDS import PlacesCNDS
import tensorflow as tf
import numpy as np
from vionaux.rnd import vidioids

class EnvronmentClassifier(object):

    def network_deployment(self, model,batch_generator, batch_size,image_size) :
        test_data = tf.placeholder(tf.float32, shape=(batch_size, 224, 224, 3))
        net =PlacesCNDS({'data':test_data})

        with tf.Session() as sesh:
            net.load(model, sesh)
            try:
                while True:
                    batch, timestamp = next(batch_generator)
                    print batch.shape
                    assert batch.shape[3] == 3
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
    VHH = vidioids.VionVideoHandler()
    batch_size = 20
    image_size = (224,224)
    batch_generator = VHH.get_batches(video_path, 0.01, 4000, 6000, batch_size, image_size)
    EC = EnvronmentClassifier()
    out = EC.network_deployment('places_CNDS_model.npy', batch_generator, batch_size, image_size)
    for i, timestamp in out:
#        print i.argmax(axis=1)
#        print timestamp
        print "one batch"
if __name__ == "__main__":
    main()
