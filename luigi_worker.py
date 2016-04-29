import luigi
from environment_extractor import EnvironmentExtractor

class InputVidoeFile(luigi.ExternalTask):

    def output(self):
        return luigi.LocalTarget("/mnt/movies03/boxer_movies/tt0401855/Underworld Evolution (2006)/Underworld.Evolution.2006.720p.Brrip.x264.Deceit.mp4")

class EnvironmentLuigi(luigi.Task):

    def requires(self):
        return InputVidoeFile()

    def output(self):
        return luigi.LocalTarget('sample_output.txt')

    def run(self):
        filepath= self.input().fn
        print "#"*10
        print filepath
        EE = EnvironmentExtractor()
        result = EE.extract_feature(filepath)
        f = self.output().open('w')
        for category_probability, timestamps in result:
            f.write(str(category_probability))
            f.write(str(timestamps))
        f.close()


if __name__ == '__main__':
    luigi.run(['EnvironmentLuigi'])
