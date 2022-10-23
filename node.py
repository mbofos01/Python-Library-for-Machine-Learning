import tools as tools

class Node:
    def __init__(self,id,biasNode,outputNode,inputNode):
        self.id = id    
        self.coming = []
        self.coming_id = []
        self.coming_weights = []
        self.going = []
        self.going_id = []
        self.going_weights = []
        self.output = 0
        self.delta = 0 
        self.bias = False
        if( biasNode):
            self.bias = biasNode
            self.output = 1
        if (outputNode):
            self.outputNode = True
            self.inputNode = False
            self.hiddenNode = False
        elif(inputNode):
            self.inputNode = True
            self.outputNode = False
            self.hiddenNode = False
        else:  
            self.hiddenNode = True
            self.outputNode = False
            self.inputNode = False
        
    def attachNode(self,fromList,intoList):
        for f in fromList:
            self.coming.append(f)
            self.coming_id.append(f.id)
            for i in range(len(f.going)):
                if(self.id == f.going_id[i]):
                    self.coming_weights.append(f.going_weights[i])
                    
        for t in intoList:
            self.going.append(t)
            self.going_id.append(t.id)
            self.going_weights.append(tools.randomWeights())
            
    def __str__(self):
        return f"Node: {self.id} from: {self.coming_id}  {self.coming_weights} into: {self.going_id}  {self.going_weights} Output: {self.output} Delta: {self.delta}"
      
    def calucateOutput(self):
        if(self.bias):
            self.output = 1
            return
        value = 0
        for i in range(len(self.coming)):
            value = value + (self.coming[i].output * self.coming_weights[i]) 
        self.output = tools.sigmoid(value)
            
    def calculateDelta(self,target):
        #print("--------- Updating node:", self.id, " ---------")
        if(self.bias):
            return
        if(self.outputNode):
            #print(f"\t Delta with {self.output * ( 1 - self.output ) * (self.output - target)}")
            self.delta = self.output * ( 1 - self.output ) * (self.output - target)
        elif(self.hiddenNode):
            sub = 0 
            for i in range(len(self.going)):
                #print(f"\t Connection with {self.going_id[i]}")
                sub = sub + self.going[i].delta * self.going_weights[i]  
            self.delta = self.output * ( 1 - self.output ) * (sub)  
        #print(f"\t {self.delta}")   
                
    def updateWeights(self,LEARNING_RATE,all):
        #print("--------- Updating node:", self.id, " ---------")
        for i in range(len(self.going)):
            #print("\tGetting delta from: " , self.going[i].delta)          
            self.going_weights[i] = self.going_weights[i] - (LEARNING_RATE * self.going[i].delta * self.output)
            #print(self.going_weights[i])
            for a in range(len(all[self.going_id[i]].coming)):
                #print(all[self.going_id[i]].coming_id[a] , " " , self.id)
                if(all[self.going_id[i]].coming_id[a] == self.id): 
                    #print(f"\tUpdating {self.going_id[i]}'s weight with {a}")
                    all[self.going_id[i]].coming_weights[a] = self.going_weights[i]
            #    if(all[self.going_id[i]].coming_id[a] == self.id): 
            #        #print(f"\tUpdating {self.going_id[i]}'s weight with {a}")
            #        all[self.going_id[i]].coming_weights[a] = self.going_weights[i]
            
        