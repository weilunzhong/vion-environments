from places_CNDS import PlacesCNDS
import tensorflow as tf
import numpy as np
from vionaux.rnd import vidioids
import csv

class EnvironmentClassifier(object):

    @staticmethod
    def get_transfer_lists():
        with open("category_index/205_to_28_categories.csv") as file:
            categories = csv.reader(file, delimiter=' ', quotechar='|')
            transfer_list, transfer_name_list = [], []
            for idx, row in enumerate(categories):
                if idx %2 != 0:
                    transfer_list.append(row[0])
                else:
                    transfer_name_list.append(row[0])
            return transfer_list, transfer_name_list

    def load_image_mean(self, path):
        mean = np.load(path)
        mean = mean.transpose(1,2,0)
        return mean

    def network_deployment(self, model,batch_generator, batch_size, image_size, mean):
        test_data = tf.placeholder(tf.float32, shape=([batch_size]+list(PlacesCNDS.scale_size)+[3]))
        net =PlacesCNDS({'data':test_data})

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
    VHH = vidioids.VionVideoHandler()
    batch_size = 20
    image_size = PlacesCNDS.scale_size
    batch_generator = VHH.get_batches(video_path, 0.01, 4000, 6000, batch_size, image_size)
    EC = EnvironmentClassifier()
    mean = EC.load_image_mean("places205_mean.npy")
    out = EC.network_deployment('places_CNDS_model.npy', batch_generator, batch_size, image_size, mean)
    transfer_list, name_list = EC.get_transfer_lists()
    print len(name_list)
    for i, timestamp in out:
        assert i.shape[0] == len(timestamp)
        print [name_list[x] for x in i.argmax(axis=1)]
#        print timestamp
        print "one batch"


if __name__ == "__main__":
    main()
