from math import sqrt

class Stats:
    def __init__(self, dataset):
        self.__dataset = dataset

    def getData(self):
        return self.__dataset

    def getDatum(self, location):
        data = self.__dataset
        return data[location]

    def setDatum(self, location, other):
        self.__dataset[location] = other
        return self.__dataset[location]

    def getMean(self):
        dataSet = self.getData()
        total = 0
        for datum in range(len(dataSet)):
            theDatum = self.getDatum(datum)
            total += theDatum
        mean = total / len(dataSet)
        return mean

    def getSD(self):
        mean = self.getMean()
        dataSet = self.getData()
        totalDeviation = 0
        for datum in range(len(dataSet)):
            deviation = self.getDatum(datum) - mean
            newDev = deviation ** 2
            totalDeviation += newDev
        variance = totalDeviation / (len(dataSet) - 1)
        SD = sqrt(variance)
        return SD

    def giveStats(self):
        mean = self.getMean()
        SD = self.getSD()
        print()
        print("Mean:", round(mean, 2))
        print("Standard Deviation:", round(SD, 2))
        for i in range(-3, 4, 1):
            score = mean + (SD * i)
            print(i, "Score: ", round(score, 2))
