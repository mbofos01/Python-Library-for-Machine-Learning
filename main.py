import BackPropagation as bp

Network = bp.BackPropagationNetwork(1000,0.9,2,18,0,1,[[0,1,0,1],[0,0,1,1]],[[0,1,1,0]])
Network.train()
#Network.test(message=True)
Network.plotError()
Network.plotSuccess()