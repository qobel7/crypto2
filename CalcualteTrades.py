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
                if list[0]=='buy':
                    buyList.append(list[1])
                if list[0]=='sell':
                    sellList.append(list[1])
        arrList = [];
        if(len(sellList)<len(buyList)):
            arrList = sellList
        else:
            arrList = buyList
        for i in range(0,len(arrList)):
            result += -(float(buyList[i])-float(sellList[i]))
        print(result)

CalcualteTrades(*argv[1:])