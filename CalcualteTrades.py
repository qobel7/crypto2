from csv import DictWriter
from sys import argv
class CalcualteTrades:
    def __init__(self,fileName):
        print("start")
        self.readFromFile(fileName)
    def readFromFile(self,fileName):
        count = 0 ;
        confList=[]
        buyList=[]
        sellList=[]
        result = 0
        with open('output/'+fileName+'.csv') as f:
            for i, line in enumerate(f):             
                list = line.split(",")
                if list[1]=='y':
                    buyList.append(list[2])
                if list[1]=='sell':
                    sellList.append(list[2])
        arrList = [];
        if(len(sellList)<len(buyList)):
            arrList = sellList
        else:
            arrList = buyList
        for i in range(0,len(sellList)):
            result += -(float(buyList[i])-float(sellList[i]))
        print(result)

CalcualteTrades(*argv[1:])