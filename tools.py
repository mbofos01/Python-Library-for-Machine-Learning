from cmath import exp
import math
import random

def sigmoid(x): 
    return 1 / ( 1 + math.e**(-x))

def randomWeights():
    num = random.random()
    neg = random.random()
    if(neg < 0.5):
        neg = -1
    else:
        neg = 1
    return neg * num

def setOutputForXOR(value,expected):
    if( value < 0.09 and expected == 0):
        return 1
    if( value > 0.9 and expected == 1):
        return 1
    
    return 0

def MSE_ERROR(actual,target):
    if len(actual) != len(target):
        raise Exception("Wrong size of vectors!")
        
    sum = 0
    for i in range(len(actual)):
        sum = sum + (target[i]-actual[i])**2
        
    return sum * 0.5

def successForXor(actual,target):
    if len(actual) != len(target):
        raise Exception("Wrong size of vectors!")
    sum = 0
    for i in range(len(actual)):
        sum = sum + setOutputForXOR(actual[i],target[i])
    
    return ( sum )/ len(actual) * 100
    

        
def MAPE_ERROR(actual,target):
    if len(actual) != len(target):
        raise Exception("Wrong size of vectors!")
    
    sum = 0
    for i in range(len(actual)):
        sum = sum + (target[i]-actual[i]) / target[i]
        
    return (sum * 100) / (len(actual))

def MAPE_SUCCESS(actual,target):
    return 100 - MAPE_ERROR(actual,target)