import os


class Dataset():
    """
    A wrapper class that contains consistent dataset and vizualization

    vizualization cla
    """
    def __init__(self,TFdataset,visualization):
        self.data = TFdataset
        self.visualization = visualization
        self.data_spec = self.data.element_spec

def savefig(x,path):
    x.savefig(path)

def numerotation(i):
    return "_%s." % i

class Visualization():
    def __init__(self,picture_gen,base_folder = None, save_folder='images',savefig=savefig):
        if base_folder is None:
            base_folder = os.path.abspath('.')
        self.base_folder = base_folder
        self.save_folder = os.path.join(self.base_folder,save_folder)
        self.picture_gen = picture_gen
        self.savefig = savefig

    def picture_from_sample(self,samples, base_name='generation',extension='png'):
        print('save')
        pictures = [self.picture_gen(sample) for sample in samples]
        N = len(pictures)
        for i in range(N):
            print(os.path.join(self.save_folder, base_name + numerotation(i) + extension))
            self.savefig(pictures[i],os.path.join(self.save_folder, base_name + numerotation(i) + extension))
