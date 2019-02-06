# -*- coding: utf-8 -*-
"""
Created on Tue Dec 22 15:55:16 2015

@author: JianyuWang
"""
import sys, getopt
import random

def parseArgs(argv):
    csvFile = ''
    imageDir = ''
    outputDir = ''
    try:
        opts, args = getopt.getopt(argv,"hc:i:o:f:",
                                   ["cFile=", "iDir=", "oDir=", "numFold="])
    except getopt.GetoptError:
        print('train_test_data.py -c <csvFile> -i <imageDir> ' + \
                '-o <outputFileDirectory> -n <numFold>')
        sys.exit(2)
    for opt, arg in opts:
        if opt == '-h':
            print('train_test_data.py -c <csvFile> -i <imageDir> ' + \
                    '-o <outputFileDirectory> -n <numFold>')
            sys.exit()
        elif opt in ("-c", "--cFile"):
            csvFile = arg
        elif opt in ("-i", "--iDir"):
            imageDir = arg
        elif opt in ("-o", "--oDir"):
            outputDir = arg
        elif opt in ("-f", "--numFold"):
            numFold = arg
    print('Csv file is: ' + csvFile)
    print('Image directory is: ' + imageDir)
    print('Output direcotry is: ' + outputDir)
    print('Number of fold is: ' + numFold)
    return [csvFile, imageDir, outputDir, int(numFold)]

def parseScoreCSV(csvFile):
    dataList = []
    
    csvFile = open(csvFile, 'r')
    # skip the header row
    csvFile.readline()
    for line in csvFile:
        lineList = line.split(',')
        # do not include the \n
        dataList.append([lineList[0], float(lineList[-1][0:-1])])
    return dataList
        

def separateTrainTest(dataList, numFold):
    trainList = []
    testList = []
    rSeed = 1
    # rSeed = 2
    
    # get random separation fold
    random.seed(rSeed)
    testIdx = random.sample(xrange(len(dataList)), 
                                int(len(dataList)/numFold))
    for i in range(len(dataList)):
        if i in testIdx:
            testList.append(dataList[i])
        else:
            trainList.append(dataList[i])
        
    return [trainList, testList]

def makeTrainTestTxt(trainList, testList, imageDir, outputDir, 
        outputSuffix = None):
    trainFile = open(outputDir + '/train' + outputSuffix + '.txt', 'w')
    testFile = open(outputDir + '/test' + outputSuffix + '.txt', 'w')
    for img in trainList:
        trainFile.write(imageDir + '/' + img[0] + ' ' + str(img[1]) + '\n')
    for img in testList:
        testFile.write(imageDir + '/' + img[0] + ' ' + str(img[1]) + '\n')
    trainFile.close()
    testFile.close()

def makeCrossValidation(dataList, numFold, imageDir, outputDir):
    oldTestList = []
    leftoverDataList = dataList
    for i in range(numFold):
        [leftoverDataList, testList] = \
                separateTrainTest(leftoverDataList, numFold - i)
        # trainList is always the previous testList and the leftover 
        makeTrainTestTxt(oldTestList + leftoverDataList, testList, imageDir,
                outputDir, '_' + str(i))
        oldTestList = oldTestList + testList


if __name__ == "__main__":
    [csvFile, imageDir, outputDir, numFold] = parseArgs(sys.argv[1:])
    dataList = parseScoreCSV(csvFile)
    # To make just one test set
    # [trainList, testList] = separateTrainTest(dataList, numFold)
    # makeTrainTestTxt(trainList, testList, imageDir, outputDir)

    # To make cross-validation
    makeCrossValidation(dataList, numFold, imageDir, outputDir)
    
    
    
    
    
    
    
    
    
    
    
    