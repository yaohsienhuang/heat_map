from matplotlib import pyplot as plt
from matplotlib import cm
from tf_keras_vis.gradcam_plus_plus import GradcamPlusPlus
from tf_keras_vis.utils.model_modifiers import ReplaceToLinear
from tf_keras_vis.utils.scores import CategoricalScore
from PIL import Image


class GenHeatMap():
    '''
    init : setup and create the GradcamPlusPlus object
    start : input arrays and plt heatmap 
    save_each_layer : save each Conv2D layers as *.png
    '''
    def __init__(self, model):
        self.model=model
        self.gradcam=GradcamPlusPlus(model ,model_modifier=ReplaceToLinear(), clone=True)
    
    def start(self,array,index,layer,save=False):
        for arr in array:
            cam = self.gradcam(CategoricalScore(index), arr, penultimate_layer=layer)
            
            if arr.shape[-1]==1:
                plt.imshow(arr, cmap='gray')
            else :
                plt.imshow(img)
            heatmap = np.uint8(cm.jet(cam[0])[..., :3] * 255)
            plt.imshow(heatmap, cmap='jet', alpha=0.5)
            
            if save==True:
                plt.savefig('output/' + str(layer) + '.png')
                plt.clf()
            else :
                plt.show()
                
    def save_each_layer(self,array,index):
        print('total layer=',len(self.model.layers))
        for i in range(len(self.model.layers)):
            if isinstance(self.model.layers[i], keras.layers.Conv2D):
                self.start(array,index,i,save=True)
                print(f'layer-{i} is saved.')

def loadData(paths_list):
    imag_array = np.zeros((len(paths_list),224,224,1),dtype=np.float32)
    for i in range(len(paths_list)):
        img = cv2.imread(paths_list[i])
        img=cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
        imag_array[i,:,:,0] = (cv2.resize(img, (224, 224), interpolation = cv2.INTER_CUBIC))/255
    print(imag_array.shape)
    return imag_array
        