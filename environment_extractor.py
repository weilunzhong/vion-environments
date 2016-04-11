from environment_classifier import EnvironmentClassifier

class EnvironmentExtractor(object):

    def __init__(self):
        pass

    def extract_feature(self, file_path):
        EC = EnvironmentClassifier()
        output = EC.run_classification(file_path)
        return output

if __name__ == "__main__":
    file_path = "/mnt/movies03/boxer_movies/tt0401855/Underworld Evolution (2006)/Underworld.Evolution.2006.720p.Brrip.x264.Deceit.mp4"
    EE = EnvironmentExtractor()
    result = EE.extract_feature(file_path)
    print result
