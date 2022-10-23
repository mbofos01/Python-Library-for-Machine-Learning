import node as n
import tools
import matplotlib.pyplot as plt

class BackPropagationNetwork:
    
    def __init__(self,EPOCHS,LEARNING_RATE,INPUTS,HIDDEN_ONE,HIDDEN_TWO,OUTPUTS,INPUT_VALUES,OUTPUT_VALUES):
        self.EPOCHS = EPOCHS
        self.LEARNING_RATE = LEARNING_RATE
        self.INPUTS = INPUTS
        self.HIDDEN_ONE = HIDDEN_ONE
        self.HIDDEN_TWO = HIDDEN_TWO
        self.OUTPUTS = OUTPUTS
        
        self.INPUT_VALUES = INPUT_VALUES
        self.OUTPUT_VALUES = OUTPUT_VALUES
        self.input_layer = []
        self.hidden_one_layer = []
        self.hidden_one_layer_non_bias = []
        self.hidden_two_layer = []
        self.hidden_two_layer_non_bias = []
        self.output_layer = []
        self.all_nodes = []
        self.TRAINING_ERROR = []
        self.TRAINING_SUCCESS = []
        self.TESTING_ERROR = []
        self.TESTING_SUCCESS = []
        self.EPOCH_LIST = []
        node_counter = 0
        
        for i in range(INPUTS):
            node = n.Node(node_counter,False,False,True)
            self.input_layer.append(node)
            self.all_nodes.append(node)
            node_counter = node_counter + 1
            
        node = n.Node(node_counter,True,False,True)
        self.input_layer.append(node)
        self.all_nodes.append(node)
        node_counter = node_counter + 1
        
        for i in range(HIDDEN_ONE):
            node = n.Node(node_counter,False,False,False)
            self.hidden_one_layer.append(node)
            self.hidden_one_layer_non_bias.append(node)
            self.all_nodes.append(node)
            node_counter = node_counter + 1
            
        node = n.Node(node_counter,True,False,False)
        self.hidden_one_layer.append(node)
        self.all_nodes.append(node)
        node_counter = node_counter + 1
        
        for i in range(HIDDEN_TWO):
            node = n.Node(node_counter,False,False,False)
            self.hidden_two_layer.append(node)
            self.hidden_two_layer_non_bias.append(node)
            self.all_nodes.append(node)
            node_counter = node_counter + 1
            
        if (HIDDEN_TWO > 0):
            node = n.Node(node_counter,True,False,False)
            self.hidden_two_layer.append(node)
            self.all_nodes.append(node)
            node_counter = node_counter + 1
        
        
        for i in range(OUTPUTS):
            node = n.Node(node_counter,False,True,False)
            self.output_layer.append(node)
            self.all_nodes.append(node)
            node_counter = node_counter + 1
            
            
        for i in range(len(self.input_layer)):
            self.input_layer[i].attachNode(intoList=self.hidden_one_layer_non_bias,fromList=[])
        for i in range(len(self.hidden_one_layer)):
            if(HIDDEN_TWO == 0):
                if(self.hidden_one_layer[i].bias == True):
                    self.hidden_one_layer[i].attachNode(intoList=self.output_layer,fromList=[])
                else:
                    self.hidden_one_layer[i].attachNode(intoList=self.output_layer,fromList=self.input_layer)
            else:
                if(self.hidden_one_layer[i].bias == True):
                    self.hidden_one_layer[i].attachNode(intoList=self.hidden_two_layer_non_bias,fromList=[])
                else:
                    self.hidden_one_layer[i].attachNode(intoList=self.hidden_two_layer_non_bias,fromList=self.input_layer)
                
        for i in range(len(self.hidden_two_layer)):
            if( self.hidden_two_layer[i].bias == True):
                self.hidden_two_layer[i].attachNode(intoList=self.output_layer,fromList=[])
            else:
                self.hidden_two_layer[i].attachNode(intoList=self.output_layer,fromList=self.hidden_one_layer)
        for i in range(OUTPUTS):
            if(HIDDEN_TWO == 0):
                self.output_layer[i].attachNode(intoList=[],fromList=self.hidden_one_layer)
            else:
                self.output_layer[i].attachNode(intoList=[],fromList=self.hidden_two_layer)

    def topology(self):
        print(f"INPUT LAYER: {self.input_layer}")
        print(f"HIDDEN 1 LAYER: {self.hidden_one_layer}")
        print(f"HIDDEN 2 LAYER: {self.hidden_two_layer}")
        print(f"OUTPUT LAYER: {self.output_layer}\n")
        
        for node in self.all_nodes:
            print(node)
   
    def train(self):

        for epoch in range(self.EPOCHS):
            self.EPOCH_LIST.append(epoch)
            er = sc = 0
            for i in range(len(self.OUTPUT_VALUES[0])):
                for input_node in range(len(self.input_layer) - 1):
                    self.input_layer[input_node].output = self.INPUT_VALUES[input_node][i]
                    
               # # Forward Pass                
                for node in self.hidden_one_layer:
                    node.calucateOutput()

                for node in self.hidden_two_layer:
                    node.calucateOutput()

                outputs = []
                targets = []
                for node in self.output_layer:
                    node.calucateOutput()
                    outputs.append(node.output)
                    
                # Calculate Deltas
                for output_node in range(self.OUTPUTS):
                    self.output_layer[output_node].calculateDelta(self.OUTPUT_VALUES[output_node][i])
                    targets.append(self.OUTPUT_VALUES[output_node][i])
                    
                                
                er = er + tools.MSE_ERROR(outputs,targets)
                sc = sc + tools.successForXor(outputs,targets)
                        

                for node in self.hidden_two_layer:
                    node.calculateDelta(None)

                for node in self.hidden_one_layer:
                    node.calculateDelta(None)

                # Update Weights
                for node in self.input_layer:
                    node.updateWeights(self.LEARNING_RATE,self.all_nodes)

                for node in self.hidden_one_layer:
                    node.updateWeights(self.LEARNING_RATE,self.all_nodes)

                for node in self.hidden_two_layer:
                    node.updateWeights(self.LEARNING_RATE,self.all_nodes)
                    
            self.TRAINING_ERROR.append(er/4)
            self.TRAINING_SUCCESS.append(sc/4)  
            
            if(epoch ==self.EPOCHS - 1 ):
                self.test(message=True)
            else:    
                self.test(message=False)
            
    def test(self,message):
        if(message):
            print("-------------------------------------------------------------")
        sc = er = 0
        for i in range(len(self.OUTPUT_VALUES[0])):
            for input_node in range(len(self.input_layer) - 1):
                self.input_layer[input_node].output = self.INPUT_VALUES[input_node][i]
                
            outputs = []
            targets = []
            # Forward Pass
            for node in self.hidden_one_layer:
                node.calucateOutput()

            for node in self.hidden_two_layer:
                node.calucateOutput()

            for node in self.output_layer:
                node.calucateOutput()
                outputs.append(node.output)
                
            for output_node in range(self.OUTPUTS):
                targets.append(self.OUTPUT_VALUES[output_node][i])

            er = er + tools.MSE_ERROR(outputs,targets)
            sc = sc + tools.successForXor(outputs,targets)
                    
            if(message):
                print(f"Value 1: {self.INPUT_VALUES[0][i]} Value 2: {self.INPUT_VALUES[1][i]} Expected: {self.OUTPUT_VALUES[0][i]} Real: {self.output_layer[0].output}")
                print("-------------------------------------------------------------")
        self.TESTING_ERROR.append(er/4)
        self.TESTING_SUCCESS.append(sc/4)
    
    def plotSuccess(self):
        fig = plt.figure()
        plt.Axes.set_frame_on
        plt.title("Success - Epoch")
        plt.plot(self.EPOCH_LIST,self.TRAINING_SUCCESS, color = 'lightskyblue')
        plt.plot(self.EPOCH_LIST, self.TESTING_SUCCESS, color = 'mediumslateblue' )

        plt.ylabel('success')
        plt.xlabel('epoch')
        plt.grid(True)
        plt.legend(["Train Data", "Test Data"], loc='best')
        plt.show()
        
    def plotError(self):
        fig = plt.figure()
        plt.Axes.set_frame_on
        plt.title("Error - Epoch")
        plt.plot(self.EPOCH_LIST,self.TRAINING_ERROR, color = 'lightskyblue')
        plt.plot(self.EPOCH_LIST, self.TESTING_ERROR, color = 'mediumslateblue' )

        plt.ylabel('error')
        plt.xlabel('epoch')
        plt.grid(True)
        plt.legend(["Train Data", "Test Data"], loc='best')
        plt.show()