import numpy as np

class Params(object):
    def __init__(self):
        self.data = {}
        self.data['numLocs'] = 100

        self.data['spaceLocsIdx'] = np.arange(0,self.data['numLocs']).conj().T
        self.data['spaceLocs'] = np.arange(0, self.data['numLocs']).conj().T
        
        self.data['samplingTime'] = 0.5
        self.data['startTime'] = 0
        self.data['endTime'] = 10
                
        self.data['noiseStd'] = 0.5

        self.data['kernel'] = {}

        self.data['kernel']['space'] = {}
        self.data['kernel']['space']['type'] = 'exponential'
        self.data['kernel']['space']['scale'] = 1
        self.data['kernel']['space']['std'] = 1

        self.data['kernel']['time'] = {}
        self.data['kernel']['time']['type'] = 'exponential' #'exponential', 'gaussian', 'periodic'
        self.data['kernel']['time']['scale'] = 1            # NOTE: to use gaussian kernel with GPKF
        self.data['kernel']['time']['std'] = 1              # scale and std must be set to 1
        self.data['kernel']['time']['frequency'] = 1
        
        # NONPERAMETRIC KERNEL parameter
        #self.np.kernel = self.data.kernel
        
        # GPKF parameters
        self.gpkf = {}
        self.gpkf['kernel'] = self.data['kernel']

        # Compute additional (common) parameters
        self.data['spaceLocsMeasIdx'] = np.sort(np.random.choice(self.data['spaceLocsIdx'], int(np.around(0.8*self.data['numLocs'])), replace=False))
        #print(self.data['spaceLocs'][self.data['spaceLocsMeasIdx']])

        self.data['spaceLocsMeas'] = self.data['spaceLocs'][self.data['spaceLocsMeasIdx']]
        self.data['spaceLocsPredIdx'] = np.setdiff1d(self.data['spaceLocsIdx'], self.data['spaceLocsMeasIdx'])
        self.data['spaceLocsPred'] = self.data['spaceLocs'][self.data['spaceLocsPredIdx']]

        """ REMEMBER THAT THAT THIS IS DIFFERENT! ORIGINAL:
        Params.data.spaceLocsMeasIdx = sort(datasample(Params.data.spaceLocsIdx, round(0.8*Params.data.numLocs), 'Replace', false));
        Params.data.spaceLocsMeas = Params.data.spaceLocs(Params.data.spaceLocsMeasIdx,:);
        Params.data.spaceLocsPredIdx = setdiff(Params.data.spaceLocsIdx, Params.data.spaceLocsMeasIdx);
        Params.data.spaceLocsPred = Params.data.spaceLocs(Params.data.spaceLocsPredIdx,:);
        """

        self.data['timeInstants'] = np.arange(self.data['startTime'], self.data['endTime'] + self.data['samplingTime'], self.data['samplingTime']).conj().T

        if self.gpkf['kernel']['time']['type'] == 'exponential':
            self.gpkf['kernel']['time']['num'] = np.sqrt(2*self.gpkf['kernel']['time']['scale'] / self.gpkf['kernel']['time']['std'])
            self.gpkf['kernel']['time']['den'] = 1/self.gpkf['kernel']['time']['std']
            
        elif self.gpkf['kernel']['time']['type'] == 'periodic':
            self.gpkf['kernel']['time']['num'] = np.sqrt(2*self.gpkf['kernel']['time']['scale'] / self.gpkf['kernel']['time']['std']) * np.array([np.sqrt((1/self.gpkf['kernel']['time']['std'])**2 + (2*pi*self.gpkf['kernel']['time']['frequency'])**2) , 1])
                    
            self.gpkf['kernel']['time']['den'] = np.array([((1/self.gpkf['kernel']['time']['std'])**2 + (2*pi*self.gpkf['kernel']['time']['frequency'])**2 ), 2/self.gpkf['kernel']['time']['std']])
            
        else:
            print('Not admissible kernel type')


"""
        elif self.gpkf['kernel']['time']['type'] == 'gaussian':
            self.gpkf['kernel']['time']['ssDim'] = 6
            load(strcat('./data/gaussian_time_kernel_approximations/ssDim=', num2str(self.gpkf['kernel']['time']['ssDim']),'_for_scale=1_std=1.mat'))
            self.gpkf['kernel']['time']['num'] = num
            self.gpkf['kernel']['time']['den'] = den
            clear num den
        """