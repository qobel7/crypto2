from Operation import Operation
import json
from csv import DictWriter
from datetime import datetime
from time import sleep
class Main:

    def start(self,confFile):
        #limit_buy(exchange_id, api_key, api_secret, pair, 100000)
        #limit_sell(exchange_id, api_key, api_secret, pair, 30)
        conf = self.readConfFile(confFile)
        
        Operation().start(conf,confFile)
    


    def readConfFile(self,confFile):
        confList=[]
        with open("conf/"+confFile+'.json') as json_file:
            data = json.load(json_file)
        return data;