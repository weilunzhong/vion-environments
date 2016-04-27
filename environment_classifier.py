from model.places_CNDS import PlacesCNDS
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
        return mean

    def network_deployment(self, model, batch_generator,
                           batch_size, image_size):
        shape = ([batch_size, image_size[1], image_size[0], 3])
        test_data = tf.placeholder(tf.float32, shape=shape)
        net =PlacesCNDS({'data':test_data})

        with tf.Session() as sesh:
            net.load(model, sesh)
            try:
                while True:
                    batch, timestamp = next(batch_generator)
                    # print "shape of the batch:", batch.shape
                    assert batch.shape[3] == 3
                    # TODO (Weilun) image mean resizing to match the dimension
                    # of input data
                    # for idx in range(0, len(batch)):
                    #     batch[idx] = np.subtract(batch[idx], mean)
                    frame_in_batch = batch.shape[0]
                    if frame_in_batch != batch_size:
                        temp_batch = np.ndarray(shape)
                        temp_batch[0:frame_in_batch] = batch
                        batch = temp_batch
                        output = sesh.run(net.get_output(),
                                          feed_dict= {test_data:batch})
                        yield output[0:frame_in_batch], timestamp
                    else:
                        output = sesh.run(net.get_output(),
                                          feed_dict= {test_data:batch})
                        yield output, timestamp
                    # graph = tf.get_default_graph()
                    # graph_def = graph.as_graph_def()
                    # print "graph_def byte size: ", graph_def.ByteSize()
                    # graph_def_s = graph_def.SerializeToString()

                    # save_path = 'PlacesCNDS.tfmodel'
                    # with open(save_path, 'wb') as f:
                    #     f.write(graph_def_s)

                    # print 'model saved'
                    # return
            except StopIteration:
                    return

    def run_classification(self, path, batch_size=20, sample_rate=0.01):
        VHH = vidioids.VionVideoHandler()
        image_width = PlacesCNDS.scale_size[0]
        params = VHH.get_video_params(path)
        ratio = float(params["width"])/params["height"]
        image_size = (int(ratio*image_width), image_width)
        batch_generator = VHH.get_batches(path, sample_rate, 0, None,
                                          batch_size, image_size)
        output = self.network_deployment('model/places_CNDS_model.npy',
                                    batch_generator, batch_size, image_size)
        # transfer_list, name_list = self.get_transfer_lists()
        # for i, timestamp in output:
        #     assert i.shape[0] == len(timestamp)
        #     print [name_list[x] for x in i.argmax(axis=1)]
        #     print timestamp
        #     print "#"*10
        return output

def main():
    video_path = "/mnt/movies03/boxer_movies/tt0401855/Underworld Evolution (2006)/Underworld.Evolution.2006.720p.Brrip.x264.Deceit.mp4"
    batch_size = 20
    EC = EnvironmentClassifier()
    EC.run_classification(video_path, batch_size, 0.01)


if __name__ == "__main__":
    main()
