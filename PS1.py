import numpy as np
import matplotlib
from matplotlib import pyplot as plt

#   This code written by ELIZABETH LAMM (elamm@nyu.edu)
#   For course DS-GA 1003 (Machine Learning & Computational Statistics)
#   Professor David Sontag, Spring 2014

def createTotalVocab(datalist):
# This function takes "datalist" of emails and returns a list of all words in the emails   
    datalen = len(datalist)
    vocabset = set()
    for i in range(0, datalen):
        thisEmailLen = len(datalist[i])
        for j in range(1, thisEmailLen):
            vocabset.add(datalist[i][j])
    vocablist = list(vocabset)
    return vocablist

def createNarrowedVocab(datalist, allVocab, minEmails):
# This function takes "datalist" of emails, "allVocab" list of all words used in 
# datalist emails, and returns a list of words that are in at least minEmails
    datalen = len(datalist)
    vocablen = len(allVocab)
    vocabBOWarray = np.zeros((datalen, vocablen))
    thisEmailVocab = set()
    for i in range(0, datalen):
        thisEmailLen = len(datalist[i])
        for j in range(1, thisEmailLen):
            thisEmailVocab.add(datalist[i][j])
        for k in range(0, vocablen):
            if allVocab[k] in thisEmailVocab:
                vocabBOWarray[i][k] = 1
            else:
                next
        thisEmailVocab.clear()
    
    allVocabCount = vocabBOWarray.sum(axis = 0, keepdims=True)
    narrowedVocab = list()
    for k in range(0, vocablen):
        if allVocabCount[0][k] >= minEmails:
            narrowedVocab.append(allVocab[k])
        else:
            next
    
    return narrowedVocab
    
def featurize(datalist, narrowedVocab):
# This function takes "datalist" of emails, "narrowedVocab" list of words used in 
# a minimum number of emails, and returns a featurized data array
    datalen = len(datalist)
    vocablen = len(narrowedVocab)
    vocabBOWarray = np.zeros((datalen, vocablen))
    thisEmailVocab = set()
    for i in range(0, datalen):
        thisEmailLen = len(datalist[i])
        for j in range(1, thisEmailLen):
            thisEmailVocab.add(datalist[i][j])
        for k in range(0, vocablen):
            if narrowedVocab[k] in thisEmailVocab:
                vocabBOWarray[i][k] = 1
            else:
                next
        thisEmailVocab.clear()
    
    return vocabBOWarray    

def labelize(datalist):
#   This function takes "datalist" of emails and returns an array of the labels,
#   with 0 labels converted to -1

    datalen = len(datalist)
    labels = np.zeros((datalen,), dtype=int)
    
    for i in range(0, datalen):
        labels[i] = 2*int(datalist[i][0])-1
        
    return labels
    
#   0. Read in data file
trainTotal = list()
trainingFile = open('spam_train.txt')
for line in trainingFile:
    trainTotal.append(line.replace('\n', '').split(' '))

#   1. Split overall training set into training set (subTrain) and validate set
splitPoint = 4000
numTrainTotal = len(trainTotal)
subTrainSet = trainTotal[0:splitPoint]
validateSet = trainTotal[splitPoint:numTrainTotal]

#   2. Transform all the data into feature vectors
#   Find all the words that occur in the training set
allVocab = createTotalVocab(subTrainSet)
minEmails = 30
narrowedVocab = createNarrowedVocab(subTrainSet, allVocab, minEmails)
STfeaturized = featurize(subTrainSet, narrowedVocab)
VSfeaturized = featurize(validateSet, narrowedVocab)
STlabels = labelize(subTrainSet)
VSlabels = labelize(validateSet)

#   3. Implement the functions perceptron_train(data) and perceptron_test(data)
def perceptron_train(data, labels, bias=0, stop=20):
#   This function takes feature vectors in array data and the data labels in vector labels
#   and trains the data using the perceptron algorithm.  It returns the trained weight vector,
#   number of mistakes, and number of passes through the data.
    numDataPoints = data.shape[0]
    numFeatures = data.shape[1]
    dataMatrix = data
    if bias == 1:
#   add bias to features
        dataMatrix = np.insert(data, 0, 1, axis = 1)

    w_vector = np.zeros((numFeatures + bias, ), dtype = float)
    predicted_class_vector = np.zeros((numDataPoints, ), dtype = int)
    
    k_mistakes = int(x=0)
    iter = int(x=0)
        
    for i in range(0, stop):
        iter += 1
        for j in range(0, numDataPoints):
            if np.vdot(dataMatrix[j,:], w_vector) >= 0:
                predicted_class_vector[j] = 1
            else:
                predicted_class_vector[j] = -1
            if predicted_class_vector[j]*labels[j] == -1:
                k_mistakes += 1
                w_vector = w_vector + labels[j]*dataMatrix[j]
        if np.amin(np.multiply(predicted_class_vector, labels)) < 0:
            next
        else:
            break
            
    return w_vector, k_mistakes, iter
    
def perceptron_test(w_vector, data, labels):
    error_count = int(x=0)
    numDataPoints = data.shape[0]
    predicted_class_vector = np.zeros((numDataPoints, ), dtype = int)
    check_errors= np.zeros((numDataPoints, ), dtype = int)
    
    for j in range(0, numDataPoints):
        if np.vdot(data[j,:], w_vector) >= 0:
            predicted_class_vector[j] = 1
        else:
            predicted_class_vector[j] = -1
    
    check_errors = np.multiply(predicted_class_vector, labels)
    for j in range(0, numDataPoints):
        if check_errors[j] < 0:
            error_count += 1
        else:
            next

    return float(error_count)/float(numDataPoints)

#   4. Train the linear classifier using your training set/Test your implementation of
#   perceptron_test by running it with the learned parameters and the training data

w_vector, k_mistakes, iter = perceptron_train(STfeaturized, STlabels)
check_training_error = perceptron_test(w_vector, STfeaturized, STlabels)
test_error = perceptron_test(w_vector, VSfeaturized, VSlabels)
print k_mistakes
print iter

print check_training_error
print test_error

#   5. Find the 15 words with the most positive weights and the 15 words with the most
#   negative weights
def find_k_largest_weights(w_vector, k, narrowedVocab):
    num_weights = w_vector.shape[0]
    words = list()
    
    sort_array = np.sort(np.copy(w_vector))
    largest_weights = np.zeros((2*k, ), dtype = float)
    largest_weights[0:k] = sort_array[0:k]
    largest_weights[k:2*k] = sort_array[num_weights-k:num_weights]

    for i in range(0, num_weights):
        if w_vector[i] in largest_weights:
            words.append([w_vector[i], narrowedVocab[i]])
        else:
            next    
    
    return sorted(words, key = lambda words:words[0])
    
lgwg_words = find_k_largest_weights(w_vector, 15, narrowedVocab)
for i in range(0, len(lgwg_words)):
    print lgwg_words[i]

#   6. Implement the averaged perceptron algorithm
def avg_perceptron_train(data, labels, bias=0, stop=20):
#   This function takes feature vectors in array data and the data labels in vector labels
#   and trains the data using the AVERAGED perceptron algorithm.  It returns the trained weight vector,
#   number of mistakes, and number of passes through the data.
    numDataPoints = data.shape[0]
    numFeatures = data.shape[1]
    dataMatrix = data
    if bias == 1:
#   add bias to features
        dataMatrix = np.insert(data, 0, 1, axis = 1)

    w_vector = np.zeros((numFeatures + bias, ), dtype = float)
    sum_w_vectors = np.zeros((numFeatures + bias, ), dtype = float)
    num_w_vectors = int(x=0)
    predicted_class_vector = np.zeros((numDataPoints, ), dtype = int)
    
    k_mistakes = int(x=0)
    iter = int(x=0)
        
    for i in range(0, stop):
        iter += 1
        for j in range(0, numDataPoints):
            sum_w_vectors = sum_w_vectors + w_vector
            num_w_vectors += 1
            if np.vdot(dataMatrix[j,:], w_vector) >= 0:
                predicted_class_vector[j] = 1
            else:
                predicted_class_vector[j] = -1
            if predicted_class_vector[j]*labels[j] == -1:
                k_mistakes += 1
                w_vector = w_vector + labels[j]*dataMatrix[j]
        if np.amin(np.multiply(predicted_class_vector, labels)) < 0:
            next
        else:
            break
            
    return sum_w_vectors/float(num_w_vectors), k_mistakes, iter

avg_w_vector, avg_k_mistakes, avg_iter = avg_perceptron_train(STfeaturized, STlabels)
check_aptrain_error = perceptron_test(avg_w_vector, STfeaturized, STlabels)
aptest_error = perceptron_test(avg_w_vector, VSfeaturized, VSlabels)
print check_aptrain_error
print aptest_error

#   7. Run the perceptron algorithm and the averaged perceptron algorithm on smaller training sets
#   Evaluate the corresponding validation error and create a plot of validation errors as a function of N
headers = ['training size', 'validation err', 'valid err - avg', 'training err', 'train err - avg', 'iters', 'iters - avg', 'mistakes', 'mistakes - avg']
training_sizes = [100, 200, 400, 800, 2000, 4000]
outputs = np.zeros((len(headers), len(training_sizes)))
column = int(x=0)
for N in training_sizes:
    subTrainSet = trainTotal[0:N]
    STfeaturized = featurize(subTrainSet, narrowedVocab)
    STlabels = labelize(subTrainSet)
    
    w_vector, k_mistakes, iter = perceptron_train(STfeaturized, STlabels)
    
    avg_w_vector, avg_k_mistakes, avg_iter = avg_perceptron_train(STfeaturized, STlabels)
    
    outputs[0][column] = training_sizes[column]
    outputs[1][column] = perceptron_test(w_vector, VSfeaturized, VSlabels)
    outputs[2][column] = perceptron_test(avg_w_vector, VSfeaturized, VSlabels)
    outputs[3][column] = perceptron_test(w_vector, STfeaturized, STlabels)
    outputs[4][column] = perceptron_test(avg_w_vector, STfeaturized, STlabels)
    outputs[5][column] = iter
    outputs[6][column] = avg_iter
    outputs[7][column] = k_mistakes
    outputs[8][column] = avg_k_mistakes
    column += 1

for i in range(0, len(headers)):
    print headers[i], outputs[i, :]

fig, ax = plt.subplots()
ax.plot(training_sizes, outputs[1], 'bo-', label = 'Perceptron')
ax.plot(training_sizes, outputs[2], 'gD-', label = 'Averaged Perceptron')
ax.legend(loc='upper right')
plt.xlabel('size of training set')
plt.ylabel('validation error')
plt.title('Validation Error for Perceptron Algorithm by Size of Training Set')
plt.show()

#   8. Also create a plot of the number of iterations as a function of N
fig, ax = plt.subplots()
ax.set_xlim(0, 4000)
ax.set_ylim(0, 15)
ax.plot(training_sizes, outputs[5], 'bo-', label = 'Perceptron')
ax.plot(training_sizes, outputs[6], 'gD-', label = 'Averaged Perceptron')
ax.legend(loc='upper right')
plt.xlabel('size of training set')
plt.ylabel('number of iterations')
plt.title('Number of Iterations for Perceptron by Size of Training Set')
plt.show()

#   9. Functions modified above to add 4th parameter "stop"

#   10. Try various configurations (reg/avg algorithm, maximum number of iterations) on your own
#   Train on the full training set
headers = ['max iters', 'validation err', 'valid err - avg', 'training err', 'train err - avg', 'iters', 'iters - avg', 'mistakes', 'mistakes - avg']
outputs = np.zeros((len(headers), 11))
column = int(x=0)
for numiters in range(1, 12):
    subTrainSet = trainTotal[0:4000]
    STfeaturized = featurize(subTrainSet, narrowedVocab)
    STlabels = labelize(subTrainSet)
    w_vector, k_mistakes, iter = perceptron_train(STfeaturized, STlabels, stop = numiters)
    avg_w_vector, avg_k_mistakes, avg_iter = avg_perceptron_train(STfeaturized, STlabels, stop = numiters)
    
    outputs[0][column] = numiters
    outputs[1][column] = perceptron_test(w_vector, VSfeaturized, VSlabels)
    outputs[2][column] = perceptron_test(avg_w_vector, VSfeaturized, VSlabels)
    outputs[3][column] = perceptron_test(w_vector, STfeaturized, STlabels)
    outputs[4][column] = perceptron_test(avg_w_vector, STfeaturized, STlabels)
    outputs[5][column] = iter
    outputs[6][column] = avg_iter
    outputs[7][column] = k_mistakes
    outputs[8][column] = avg_k_mistakes
    column += 1
    
for i in range(0, len(headers)):
    print headers[i], outputs[i, :]
 
fig, ax = plt.subplots()
ax.set_xlim(0, 15)
ax.set_ylim(0, .05)
ax.plot(outputs[0], outputs[1], 'bo-', label = 'Perceptron - Validation Error')
ax.plot(outputs[0], outputs[2], 'gD-', label = 'Averaged Perceptron - Validation Error')
ax.plot(outputs[0], outputs[3], 'b.--', label = 'Perceptron - Training Error')
ax.plot(outputs[0], outputs[4], 'gd--', label = 'Averaged Perceptron - Training Error')
ax.legend(loc='upper right')
plt.xlabel('maximum iterations')
plt.ylabel('error')
plt.title('Error for Perceptron Algorithm by Maximum Iterations Allowed')
plt.show()    
    
trainSet = trainTotal
TSfeaturized = featurize(trainSet, narrowedVocab)
TSlabels = labelize(trainSet)
w_vector, k_mistakes, iter = avg_perceptron_train(TSfeaturized, TSlabels, stop = 5)
trainError = perceptron_test(w_vector, TSfeaturized, TSlabels)

testTotal = list()
testFile = open('spam_test.txt')
for line in testFile:
    testTotal.append(line.replace('\n', '').split(' '))
testFeaturized = featurize(testTotal, narrowedVocab)
testLabels = labelize(testTotal)
testError = perceptron_test(w_vector, testFeaturized, testLabels)

print trainError
print testError