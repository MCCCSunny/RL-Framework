import numpy as np
import random
import sys
import json
try:
    from PolicyTrainVisualize import PolicyTrainVisualize
except Exception as e:
    from tools.PolicyTrainVisualize import PolicyTrainVisualize
import os
import re

"""
    plot_meta_simulation.py <settings_file_name> <path_to_data>
    Example:
    python3 tools/plot_meta_simulation.py ../../../backup/Particle_Sim/algorithm.PPO.PPO/ ../../../backup/Particle_Sim/algorithm.PPO.PPO/Nav_Sphere_SMBAE_10D_Hyper_action_learning_rate_0.025/PPO_SMBAE_10D_Hyper.json
"""

def getDataFolderNames(prefixPath, folderPrefix, settings):
    path = prefixPath
    folder_ = folderPrefix
    name_suffix = "/"+settings["model_type"]+"/"+"trainingData_" + str(settings['agent_name']) + ".json"
    folderNames = []
    print ("path: ", path)
    print ("folder_: ", folder_)
    print ("name_suffix: ", name_suffix)
    if os.path.exists(path):
        for filename in os.listdir(path):
            if re.match(folder_ + "\d+", filename):
                folderNames.append(prefixPath + filename + name_suffix)
                print ("Folder Name: ", prefixPath + filename + name_suffix)
    ## For the case where I put a slash at the end of the path name...
    path = path + folder_[:-1]
    print ("New path: ", path)
    if os.path.exists(path):
        for filename in os.listdir(path):
            print ("Testing path: ", filename)
            print ("Testing patern: ", folder_[-1:] + "\d+")
            if re.match(folder_[-1:] + "\d+", filename):
                folderNames.append(path + filename + name_suffix)
                print ("Folder Name: ", path + filename + name_suffix)
    return folderNames
        
def plotMetaDataSimulation(data_path, settings, settingsFiles, folder=''):   
    
    rlv = PolicyTrainVisualize("Training Curves", settings=settings)
    otherDatas = []
    # settingsFiles = sys.argv[2:]
    print("settingsFiles: ", settingsFiles)
    for settingsFile_ in settingsFiles:
        print ("Loading settings file: ", settingsFile_)
        settingsFile_ = open(settingsFile_, 'r')
        settings = json.load(settingsFile_)
        settingsFile_.close()
    
        folder_ = settings['data_folder'] + "_"
        folderNames_ = getDataFolderNames(data_path, folder_, settings)
        trainingDatas = []
        # Need to train a better Baseline
        for folderName in folderNames_:
            trainData={}
            trainData['fileName']=folderName
            trainData['name']= settings['data_folder']
            # trainData['colour'] = (1.0, 0.0, 0.0, 1.0)
            trainingDatas.append(trainData)
    
        otherDatas.append(trainingDatas)
    
    
    min_length = 1000000000
    for j in range(len(otherDatas)):
        tmp_min_length = 1000000000
        for i in range(len(otherDatas[j])):
            datafile = otherDatas[j][i]['fileName']
            file = open(datafile)
            otherDatas[j][i]['data'] = json.load(file)
            print ("Data file: ", file)
            print("Length of data: ", len(otherDatas[j][i]['data']["mean_eval"]))
            if min_length > (len(otherDatas[j][i]['data']["mean_eval"])):
                min_length = len(otherDatas[j][i]['data']["mean_eval"])
            if tmp_min_length > (len(otherDatas[j][i]['data']["mean_eval"])):
                tmp_min_length = len(otherDatas[j][i]['data']["mean_eval"])
            file.close()
        print ("Min length for ", otherDatas[j][0]['fileName'], " is ", tmp_min_length)
        
    rlv.setLength(min_length)
    rlv.updateRewards(trainingDatas, otherDatas)
    rlv.init()
    rlv.saveVisual(folder+"MBAE_Training_curves")
    return rlv
