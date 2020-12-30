import numpy as np

class Params(object):
    def __init__(self, df, locations, dataset_name):
        self.dataset = dataset_name
        self.optimizer_restarts = 2
        
        if self.dataset == 'synthetic':
            self.data = {}
            self.data['numLocs'] =100

            self.data['spaceLocsIdx'] = np.arange(0,self.data['numLocs']).conj().T
            self.data['spaceLocs'] = np.arange(0, self.data['numLocs']).conj().T

            self.data['samplingTime'] = 1.0
            self.data['startTime'] = 0
            self.data['endTime'] = 10
            
            self.data['timeInstants'] = np.arange(self.data['startTime'], self.data['endTime'] + self.data['samplingTime'], self.data['samplingTime']).conj().T

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
            
        else:
            print('Using fornes dataset')
            self.data = {}
            self.data['numLocs'] = df.columns.size

            self.data['spaceLocsIdx'] = np.arange(0,self.data['numLocs']).conj().T
                    
            self.data['spaceLocs'] = locations
            
            #self.data['spaceLocs'] = np.array([3,6,9,12]).conj().T
            #self.data['spaceLocs'] = np.array([0,0,3], [0,0,6], [0,0,9], [0,0,12])
            #self.data['spaceLocs'] = np.arange(0, self.data['numLocs']).conj().T

            self.data['samplingTime'] = 1.0 #(1.0 / 24 / 60)
            self.data['startTime'] = df.index[0]
            self.data['endTime'] = df.index[-1]
            
            self.data['timeInstants'] = df.index.T

            self.data['noiseStd'] = 0.05

            self.data['kernel'] = {}

            self.data['kernel']['space'] = {}
            self.data['kernel']['space']['type'] = 'gaussian'
            self.data['kernel']['space']['scale'] = 0.001
            self.data['kernel']['space']['std'] = 2

            self.data['kernel']['time'] = {}
            self.data['kernel']['time']['type'] = 'exponential'    #'exponential', 'gaussian', 'periodic'
            self.data['kernel']['time']['scale'] = 10          # NOTE: to use gaussian kernel with GPKF
            self.data['kernel']['time']['std'] = 2            # scale and std must be set to 1
            self.data['kernel']['time']['frequency'] = 1
            
            if self.data['kernel']['time']['type'] == 'periodic':
                self.data['kernel']['time']['scale'] = 0.1        
                self.data['kernel']['time']['std'] = 2              
                self.data['kernel']['time']['frequency'] = 750

            # NONPERAMETRIC KERNEL parameter
            #self.np.kernel = self.data.kernel

            # GPKF parameters
            self.gpkf = {}
            self.gpkf['kernel'] = self.data['kernel']

        # Compute additional (common) parameters
        self.data['spaceLocsMeasIdx'] = np.sort(np.random.choice(self.data['spaceLocsIdx'], int(np.around(1.0*self.data['numLocs'])), replace=False))
        self.data['spaceLocsMeas'] = self.data['spaceLocs'][self.data['spaceLocsMeasIdx']]
        
        # Define prediction mesh
        xx, yy, zz = np.meshgrid(np.arange(0,66,6), 3, np.arange(0,66,6))
        self.data['spaceLocsPredIdx'] = np.arange(0, 11*11)
        #self.data['spaceLocsPred'] = np.column_stack([xx.flatten(), yy.flatten(), zz.flatten()])
        
        #self.data['spaceLocsPredIdx'] = np.arange(0, 53) #np.setdiff1d(self.data['spaceLocsIdx'], self.data['spaceLocsMeasIdx'])
        #self.data['spaceLocsPred'] = np.linspace(0,13,53) #self.data['spaceLocs'][self.data['spaceLocsPredIdx']]
        
        

        #self.data['timeInstants'] = np.arange(self.data['startTime'], self.data['endTime'] + self.data['samplingTime'], self.data['samplingTime']).conj().T

"""

        if self.gpkf['kernel']['time']['type'] == 'exponential':
            self.gpkf['kernel']['time']['num'] = np.array([np.sqrt(2*self.gpkf['kernel']['time']['scale'] / self.gpkf['kernel']['time']['std'])])
            self.gpkf['kernel']['time']['den'] = np.array([1/self.gpkf['kernel']['time']['std']])
            
        elif self.gpkf['kernel']['time']['type'] == 'periodic':
            self.gpkf['kernel']['time']['num'] = np.array([np.sqrt(2*self.gpkf['kernel']['time']['scale'] / self.gpkf['kernel']['time']['std']) * np.array([np.sqrt((1/self.gpkf['kernel'] ['time']['std'])**2 + (2*np.pi*self.gpkf['kernel']['time']['frequency'])**2) , 1])])
                    
            self.gpkf['kernel']['time']['den'] = np.array([((1/self.gpkf['kernel']['time']['std'])**2 + (2*np.pi*self.gpkf['kernel']['time']['frequency'])**2 ), 2/self.gpkf['kernel']['time']['std']])
            
        #elif self.gpkf['kernel']['time']['type'] == 'gaussian':
        else:
            print('Not admissible kernel type')
            
"""