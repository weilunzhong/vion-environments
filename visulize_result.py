import tensorflow as tf
import csv
import numpy as np
from model.places_CNDS import PlacesCNDS
from vionaux.rnd import vidioids
import cv2

class EnvronmentClassifier(object):

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
        shape = ([batch_size, image_size[1], image_size[0], 3])
        test_data = tf.placeholder(tf.float32, shape=shape)
        net = PlacesCNDS({'data':test_data})

        with tf.Session() as sesh:
            net.load(model, sesh)
            try:
                while True:
                    batch, timestamp = next(batch_generator)
                    print batch.shape
                    assert batch.shape[3] == 3
                    # for idx in range(0, len(batch)):
                    #     batch[idx] = np.subtract(batch[idx], mean)
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
    image_width = PlacesCNDS.scale_size[0]
    params = VHH.get_video_params(video_path)
    ratio = float(params["width"])/params["height"]
    image_size = (int(ratio*image_width), image_width)
    batch_size = 20
    image_size = PlacesCNDS.scale_size
    batch_generator = VHH.get_batches(video_path, 0.1, 1000, 2000, batch_size, image_size)
    EC = EnvronmentClassifier()
    mean = EC.load_image_mean("places205_mean.npy")
    out = EC.network_deployment('model/places_CNDS_model.npy', batch_generator, batch_size, image_size, mean)
    transfer_list, transfer_name_list = EC.get_transfer_lists()
    for i, timestamp in out:
        assert i.shape[0] == len(timestamp)
        res_name = [transfer_name_list[idx] for idx in i.argmax(axis=1)]
        print res_name
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
