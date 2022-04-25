import sys
import matplotlib.pyplot as plt
import numpy as np
import numpy.linalg as LA
import scipy.cluster.hierarchy as hierarchy

def load_data(filepath):
    with open(filepath, encoding='utf-8') as f:
        labels = f.readline().strip().split(',')
        pokemons = list()
        for line in f:
            feature = line.strip().split(',')
            pokemon = dict(zip(labels, feature))
            pokemons.append(pokemon)
        return pokemons

def calc_features(row):
    features = np.zeros((6,), dtype='int64')
    features[0] = int(row['Attack'])
    features[1] = int(row['Sp. Atk'])
    features[2] = int(row['Speed'])
    features[3] = int(row['Defense'])
    features[4] = int(row['Sp. Def'])
    features[5] = int(row['HP'])
    return features

def hac(features):

    n = len(features)
    result = np.zeros((n-1, 4))  # The  numpy array that record each cluster information
    cluster = np.array([-1]*n )  # Keep track of the current cluster situation
    # Setting up the distance matrix that keep track of the current distance between each cluster
    distance = np.zeros((n, n)) 
    for i in range(0, n):
        for j in range(0, n):
            if i != j:
                distance[i][j] = LA.norm(np.array(features[i])-np.array(features[j]))
            else:
                distance[i][j] = sys.maxsize
                
    # In each iteration, fill in one row of the result. i.e. connect two clusters 
    for i in range(0, n-1):
        # find the minimum  distance
        minDistance = np.amin(distance) 
        minSmallerIndex = sys.maxsize 
        minBiggerIndex = sys.maxsize 
        for j in range(0,n):
            for k in range(0, n):
                #Tie breaking
                if distance[j][k] == minDistance:
                    #Find the corresponding cluster index
                    if cluster[j] == -1:
                        temp1 = j
                    else: 
                        temp1 = cluster[j]+ n  
                    if cluster[k] == -1:
                        temp2 = k
                    else: 
                        temp2 = cluster[k]+ n 
                    #Sort the two index
                    if temp1 < temp2:
                        oneSmallerThanTwo= True
                    else:
                        oneSmallerThanTwo = False
                    smallerIndex = min( temp1 , temp2 ) 
                    biggerIndex = max( temp1 , temp2 ) 
                    #if find a new smallest index
                    if  smallerIndex < minSmallerIndex:
                        minSmallerIndex =  smallerIndex
                        minBiggerIndex = biggerIndex
                        if oneSmallerThanTwo:
                            row  = j
                            column = k
                        else:
                            row  = k
                            column = j
                    #if the smaller index is equivalent but the bigger index is smaller than the original one
                    elif smallerIndex == minSmallerIndex:
                        if  biggerIndex < minBiggerIndex:
                            minSmallerIndex =  smallerIndex
                            minBiggerIndex = biggerIndex
                            if oneSmallerThanTwo:
                                row  = j
                                column = k
                            else:
                                row  = k
                                column = j
                        
        # Select the two clusters
        clusterSameAsRow = list()
        clusterSameAsColumn = list()
        if cluster[row] == -1:
            clusterSameAsRow.append(row)
        else:
            clusterSameAsRow = [j for j in range(n) if cluster[j] == cluster[row]]
        if cluster[column] == -1:
            clusterSameAsColumn.append(column)
        else:
            clusterSameAsColumn = [j for j in range(n) if cluster[j] == cluster[column]]

        # record the result
        combinedCluster = clusterSameAsColumn + clusterSameAsRow
        result[i][0] = minSmallerIndex
        result[i][1] = minBiggerIndex
        result[i][2] = minDistance
        result[i][3] = len(combinedCluster)
        
        # Update the distance matrix
        for cl in clusterSameAsColumn:
            for r in clusterSameAsRow:
                distance[r][cl] = sys.maxsize
        for j in range(0, n):
            if j not in combinedCluster:
                distance[j][combinedCluster] = max([distance[j][k]for k in combinedCluster])
        for j in combinedCluster:
            for k in range(0, n):
                distance[j][k] = distance[k][j]
        cluster[combinedCluster] = i
    return result

def imshow_hac(Z):
    fig = plt.figure(figsize=(25, 10))
    dn = hierarchy.dendrogram(Z)
    plt.show()

def main():
    x = [calc_features(row) for row in load_data('Pokemon.csv')][:10]
    imshow_hac(hac(x))
    imshow_hac(hierarchy.linkage(x, method='complete'))
    
if __name__ == "__main__":
    main()
