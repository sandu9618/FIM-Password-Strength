import numpy as np
import pandas as pd
import define_


def createC1(dataSet):
    C1 = []
    for transaction in dataSet:
        for item in transaction:
            if not [item] in C1:
                C1.append([item])

    C1.sort()
    print(C1)
    return list(map(frozenset, C1))  # use frozen set so we
    # can use it as a key in a dict


def scanD(D, Ck, minSupport):  # generates L and dictionary of support data
    ssCnt = {}
    for tid in D:
        for can in Ck:
            if can.issubset(tid):
                if not can in ssCnt:
                    ssCnt[can] = 1
                else:
                    ssCnt[can] += 1
    numItems = float(len(D))
    retList = []
    supportData = {}
    for key in ssCnt:
        support = ssCnt[key] / numItems
        if support >= minSupport:
            retList.insert(0, key)
        supportData[key] = support
    return retList, supportData


def aprioriGen(Lk, k):  # creates Ck
    retList = []
    lenLk = len(Lk)
    for i in range(lenLk):
        for j in range(i + 1, lenLk):
            L1 = list(Lk[i])[:k - 2]
            L2 = list(Lk[j])[:k - 2]
            L1.sort()
            L2.sort()
            if L1 == L2:  # if first k-2 elements are equal
                retList.append(Lk[i] | Lk[j])  # set union
    return retList


def apriori(dataSet, minSupport=0.1):
    C1 = createC1(dataSet)  # Create candidate itemsets of size one
    D = list(map(set, dataSet))  # make dataset in the setform
    L1, supportData = scanD(D, C1, minSupport)
    ''' returns itemsets that meet
                minimum requirement and dictionary with support values
                '''
    L = [L1]
    k = 2
    while len(L[k - 2]) > 0:
        Ck = aprioriGen(L[k - 2], k)  # To produce candidate itemset of size k
        Lk, supK = scanD(D, Ck, minSupport)  # scan DB to get Lk
        supportData.update(supK)
        L.append(Lk)
        k += 1
    return L, supportData


def generateRules(L, supportData, minConf=0.7):  # supportData is a dict coming from scanD
    # freqt = freq(L)
    # print(pd.Series( (v[0] for v in L) ))
    bigRuleList = []
    for i in range(1, len(L)):  # only get the sets with two or more items
        for freqSet in L[i]:
            H1 = [frozenset([item]) for item in freqSet]
            if i > 1:
                rulesFromConseq(freqSet, H1, supportData, bigRuleList, minConf)
            else:
                calcConf(freqSet, H1, supportData, bigRuleList, minConf)
    return bigRuleList


def calcConf(freqSet, H, supportData, brl, minConf=0.7):
    prunedH = []  # create new list to return
    list_item = []
    for conseq in H:
        confAB = supportData[freqSet] / supportData[freqSet - conseq]  # calc
        confBA = supportData[freqSet] / supportData[conseq]
        lift = supportData[freqSet] / supportData[freqSet - conseq] * supportData[conseq]
        if (confAB >= minConf) and (
                (freqSet - conseq) in [frozenset({define_.STRENGTH_1}), frozenset({define_.STRENGTH_2}),
                                       frozenset({define_.STRENGTH_3}), frozenset({define_.STRENGTH_4}),
                                       frozenset({define_.STRENGTH_5})]):
            print(freqSet - conseq, '-->', conseq, 'confAB:', confAB, 'confBA:', confBA, 'supportAB:',
                  supportData[freqSet], 'supportA:', supportData[freqSet - conseq], 'supportB:', supportData[conseq],
                  'lift:', lift)
            brl.append((list(freqSet - conseq), list(conseq), confAB, confBA, supportData[freqSet],
                        supportData[freqSet - conseq], supportData[conseq], lift))
            prunedH.append(conseq)
    return prunedH


def rulesFromConseq(freqSet, H, supportData, brl, minConf=0.7):
    m = len(H[0])
    if len(freqSet) > (m + 1):  # try further merging
        Hmp1 = aprioriGen(H, m + 1)  # create Hm+1 new candidates
        Hmp1 = calcConf(freqSet, Hmp1, supportData, brl, minConf)
        if len(Hmp1) > 1:  # need at least two sets to merge
            rulesFromConseq(freqSet, Hmp1, supportData, brl, minConf)


def convertToStringList(string):
    a = string.replace('\'', '')
    b = a.replace('[', '')
    c = b.replace(']', '')
    l = c.split(", ")
    return l


def oneHot(dataSet, featureList):
    r, c = dataSet.shape
    zeroArray = np.zeros(shape=(r, len(featureList)))
    for index, row in dataSet.iterrows():
        ItemA = row['antecedents']
        ItemB = row['consequents']

        for item in ItemA:
            if item in featureList:
                indexA = featureList.index(item)
                zeroArray[index][indexA] = 1

        if ItemB[0] in featureList:
            indexB = featureList.index(ItemB[0])
            zeroArray[index][indexB] = 1

    return zeroArray


def freqItemToDF(freq, cols):
    freqArr = []
    for i in range(len(freq) - 1):
        len_ = len(freq[i])
        for index, item in freq[i]:
            arr = np.array(
                ['*     ', '*     ', '*     ', '*         ', '*       ', '*      ', '*      ', '*      ', '*      '])
            # item = list(freq[i][j])
            for k in range(len(item)):
                kItem = item[k]
                if 'a_' in kItem:
                    arr[0] = kItem
                elif 'A_' in kItem:
                    arr[1] = kItem
                elif '0_' in kItem:
                    arr[2] = kItem
                elif 'sy' in kItem:
                    arr[3] = kItem
                elif 'le' in kItem:
                    arr[4] = kItem
                elif 'is_l' in kItem:
                    arr[5] = kItem
                elif 'is_u' in kItem:
                    arr[6] = kItem
                elif 'is_n' in kItem:
                    arr[7] = kItem
                elif 'st' in kItem:
                    arr[8] = kItem

            freqArr.append(arr)

    freqDF = pd.DataFrame(freqArr, columns=cols)

    return freqDF
