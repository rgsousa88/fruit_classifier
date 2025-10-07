import os, sys
import json

class ConfigParser():
    def __init__(self, configPath:str):
        self.configFilePath = configPath
        self.config = self.load()
    
    def load(self,):
        with open(self.configFilePath,'r') as file:
            config = json.load(file)
        
        return config
    
if __name__ == "__main__":
    configFile = "config.json"
    parser = ConfigParser(configPath=configFile)
    print(parser.config)
             