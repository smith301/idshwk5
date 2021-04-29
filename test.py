from sklearn.ensemble import RandomForestClassifier
import math


class Domain:
    def __init__(self, name, label=None):
        self.name = name.strip()
        self.length = len(name)
        if label:
            self.label = label.strip()
        else:
            self.label = None
        self.entropy = Domain.calEntropy(self.name)

    def getFeature(self):
        return [self.length, self.entropy]
        #return [self.entropy]

    @staticmethod
    def calEntropy(text):
        entropy = 0
        charDict = {}
        for char in text:
            if char not in charDict:
                charDict[char] = 1
            else:
                charDict[char] += 1
        for key in charDict:
            p = charDict[key] / len(text)
            entropy += -(p * math.log2(p))
        return entropy


def getData(fileName, domainList):
    with open(fileName, 'r') as f:
        for i in f.readlines(1024 * 1000):
            if ',' in i:
                name, label = i.split(',')
            else:
                name = i
                label = None
            domainList.append(Domain(name, label))


if __name__ == "__main__":
    domainList = []
    testDomainList = []
    featureMetrix = []
    labelMetrix = []
    testMetrix = []

    getData('train.txt', domainList)
    getData('test.txt', testDomainList)

    for domain in domainList:
        featureMetrix.append(domain.getFeature())
        labelMetrix.append(domain.label)

    clf = RandomForestClassifier(random_state=0)
    clf.fit(featureMetrix, labelMetrix)

    with open('result.txt', 'w') as f:
        for domain in testDomainList:
            f.write(domain.name)
            f.write(',')
            f.write(clf.predict([domain.getFeature()])[0])
            f.write('\n')
