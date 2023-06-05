#NAME: Kerlous Aweeda
#HW1 Part II: Python program solving the questions and check report.pdf to know how to run the program

import math

def main():
        #PRE-PROCESSING MODELLING
	# unigram maximum likelihood model
	unigramWordCount = {} # all unique word tokens in the training corpus
	initializeWordCount("train.txt", unigramWordCount, "unigram") 
	# pre-processing: add <s> and </s> and <unk> Train Corpus
	processFile("train.txt",  unigramWordCount)

	# pre-processing: add <unk> Test Corpus
	modifyTestFile("test.txt", unigramWordCount)
	
	allWordsTrainingCorpus = int(totalCountOf(unigramWordCount)) # all word (NOT UNIQUE) tokens training corpus

	# bigram maximum likelihood model (will also be used for add-one)
	bigramWordCount = {} # all bigram word count of train corpus
	initializeWordCount("processed-train.txt", bigramWordCount, "bigram")
	print("-----------------------------------------------------------------------------")
	# Question 1
	print("PROBLEM 1")
	print("Vocabulary size: ", vocabularySize(unigramWordCount)) #return unique word tokens in the training corpus
	print("-----------------------------------------------------------------------------")

	# Question 2
	print("PROBLEM 2")
	print("Number of tokens: " , allWordsTrainingCorpus)
	print("-----------------------------------------------------------------------------")
	
	# Question 3 (using unmodified test.txt)
	print("PROBLEM 3")
	print("% of missing word tokens and word types from test corpus: ",
		percentageMissingUnigram("test.txt", unigramWordCount),"%")
	print("-----------------------------------------------------------------------------")

	
	# Question 4
	print("PROBLEM 4")
	print("%  of missing bigram types from test corpus: ",
		percentageMissingBigram("test.txt", bigramWordCount),"%")
	print("-----------------------------------------------------------------------------")
	

	# Question 5
	print("PROBLEM 5")
	data = " <s> i look forward to hearing your reply . </s>"
	print("Log Probability:" + "\"" + data + "\""+'\n')
	print("Word/Tuple: log probability")
	print("Unigram:", logUnigramProbability5(data, unigramWordCount, allWordsTrainingCorpus),'\n')
	print("Word/Tuple: log probability")
	print("Bigram:", logBigramProbability5(data, unigramWordCount, bigramWordCount),'\n')
	print("Word/Tuple: log probability")
	print("Bigram Add-One:", logBigramAddOneProbability5(data, unigramWordCount, bigramWordCount))
	print("-----------------------------------------------------------------------------")

	# Question 6
	print("PROBLEM 6")
	print("Perplexity of: " + "\"" + data + "\"")
	print("Unigram:", perplexity("unigram", data, unigramWordCount, bigramWordCount, allWordsTrainingCorpus),'\n')
	print("Bigram:", perplexity("bigram", data, unigramWordCount, bigramWordCount, allWordsTrainingCorpus),'\n')
	print("Bigram Add-One:", perplexity("bigram add one", data, unigramWordCount, bigramWordCount, allWordsTrainingCorpus))
	print("-----------------------------------------------------------------------------")
	
	# Question 7
	print("PROBLEM 7")
	corpus = ""
	with open("processed-test.txt", "r",encoding="utf8") as file:
		for line in file:
			line=line.replace("<\s>" ,"<\s> <end>")
			corpus +=  line
					
			
	print("Perplexity of Entire Test Corpus")
	print("Unigram: ",perplexity("unigram", corpus, unigramWordCount, bigramWordCount, allWordsTrainingCorpus))
	print("Bigram: ", perplexity("bigram", corpus, unigramWordCount, bigramWordCount, allWordsTrainingCorpus))
	print("Bigram Add-One: ", perplexity("bigram add one", corpus, unigramWordCount, bigramWordCount, allWordsTrainingCorpus))

      

# Reads the input file and initializes wordCount
def initializeWordCount(inputFile, wordCount, whichModel):
	with open(inputFile, encoding="utf8") as file:
		for line in file:
			line = line.lower()
			if whichModel == "unigram":
				for word in line.split():
					if word not in wordCount:
						wordCount[word] = 1
					else:
						wordCount[word] += 1
			else: 
				lineAsList = line.split()
				for i in range(len(lineAsList) - 1):
					wordTuple = (lineAsList[i], lineAsList[i+1])
					if wordTuple not in wordCount:
						wordCount[wordTuple] = 1
					else:
						wordCount[wordTuple] += 1



# Reads TRAIN corpus file, then create modified input file with pad symbols and <unk> tokens 
def processFile(inputFile, wordCount):
	outputFile = open("processed-" + inputFile, "w",encoding="utf8")
	wordCount["<unk>"] = 0
	wordCount["<s>"] = 0
	wordCount["</s>"] = 0
	with open(inputFile, "r",encoding="utf8") as inFile:
		for line in inFile:
			line = line.lower()
			wordCount["<s>"] += 1
			outputFile.write("<s>")
			for word in line.split():
				if wordCount[word] == 1:
					outputFile.write(" <unk>")
					wordCount["<unk>"] += 1
					del wordCount[word]
				else:
					outputFile.write(" " + word)
			wordCount["</s>"] += 1
			outputFile.write(" </s>\n")
	outputFile.close()


# Replaces all words seen in TEST corpus file not seen in training with <unk>, then add pad symbols
def modifyTestFile(testFile, wordCount) :
	outputFile = open("processed-" + testFile, "w",encoding="utf8")
	with open(testFile, "r",encoding="utf8") as inFile:
		for line in inFile:
			line = line.lower()
			outputFile.write("<s>")
			for word in line.split():
				if word not in wordCount:
					outputFile.write(" <unk>")
				else:
					outputFile.write(" " + word)
			outputFile.write(" </s>\n")
	outputFile.close()

# Returns total count of word tokens
def totalCountOf(wordCount):
	totalCount = 0.0
	for count in wordCount.values():
		totalCount += count
	return totalCount

# Returns word types count
def vocabularySize(unigramWordCount):
	count = 0
	for word in unigramWordCount:
		count += 1
	return count

# testing word count
def printWordCount(wordCount):
	for word, count in wordCount.items():
		print(word, count)


# solving for percentage of missing words 
def percentageMissingUnigram(testFile, wordCount):
	countMissing = 0.0
	countTotal = 0.0
	with open(testFile, "r",encoding="utf8") as inFile:
		for line in inFile:
			line = line.lower()
			for word in line.split():
				countTotal += 1
				if word not in wordCount:
					countMissing += 1
	return '%.3f'%(countMissing / countTotal * 100)

# solving for percentage of missing bigram types
def percentageMissingBigram(testFile, wordCount):
	countMissing = 0.0
	countTotal = 0.0
	with open(testFile, "r",encoding="utf8") as inFile:
		for line in inFile:
			lineAsList = line.split()
			for i in range(len(lineAsList) - 1):
				countTotal += 1
				wordTuple = (lineAsList[i], lineAsList[i+1])
                #if bigram word token of test corpus is not found in training corpus bigram word dictionary
				if wordTuple not in wordCount: 
                     #increments every time a missing bigram types is found
					countMissing += 1 
	return '%.3f'%(countMissing/countTotal * 100)


# solving unigram probability
def unigramProbability(sentence, wordCount, size):
	prob = 1.0
	for word in sentence.split():
		prob *= (wordCount[word] / size)
	return prob

# problem 5 unigram probability w/ parameters
def unigramProbability5(sentence, wordCount, size):
	prob = 1.0
	for word in sentence.split(): 
		print(word,":  ",'%.3f'%(math.log(wordCount[word]/size,2)))
		prob *= (wordCount[word] / size)
	return prob

# solving unigram log probability
def logUnigramProbability(sentences, wordCount, size):
	prob = 0.0
	for line in sentences.split("\n"):		
		sentenceProb = unigramProbability(line, wordCount, size)
		if sentenceProb == 0: 
			return "undefined"
		else:
			prob += math.log(sentenceProb, 2)
	return prob

# problem 5 log unigram probability w/ parameters
def logUnigramProbability5(sentences, wordCount, size):
	prob = 0.0
	for line in sentences.split("\n"):		
		sentenceProb = unigramProbability5(line, wordCount, size)
		if sentenceProb == 0: 
			return "undefined"
		else:
			prob += math.log(sentenceProb, 2)
	return '%.3f'%(prob)


# solving bigram probability
def bigramProbability(sentence, unigramWordCount, bigramWordCount):
	lineAsList = sentence.split()
	prob = 1.0
	for i in range(len(lineAsList) - 1):
		wordTuple = (lineAsList[i], lineAsList[i+1])
		if wordTuple not in bigramWordCount:
			return 0
		prob *= (bigramWordCount[wordTuple] / float(unigramWordCount[lineAsList[i]]))
	return prob

# problem 5 bigram probability w/ parameters
def bigramProbability5(sentence, unigramWordCount, bigramWordCount):
	lineAsList = sentence.split()
	prob = 1.0
	for i in range(len(lineAsList) - 1):
		wordTuple = (lineAsList[i], lineAsList[i+1])
		if wordTuple not in bigramWordCount:
			print(wordTuple," has undefined log probability.")
			return 0
		print(wordTuple,":  ",'%.3f'%(math.log(bigramWordCount[wordTuple]/float(unigramWordCount[lineAsList[i]]),2)))
		prob *= (bigramWordCount[wordTuple] / float(unigramWordCount[lineAsList[i]]))
	return prob

# solving bigram log probability
def logBigramProbability(sentences, unigramWordCount, bigramWordCount):
	prob = 0.0
	for line in sentences.split("\n"):
		sentenceProb = bigramProbability(line, unigramWordCount, bigramWordCount)
		if sentenceProb == 0: 
			return "undefined"
		else: prob += math.log(sentenceProb, 2)
	return prob

# problem 5 log bigram probability w/ parameters
def logBigramProbability5(sentences, unigramWordCount, bigramWordCount):
	prob = 0.0
	for line in sentences.split("\n"):
		sentenceProb = bigramProbability5(line, unigramWordCount, bigramWordCount)
		if sentenceProb == 0: 
			return "undefined"
		else:
			prob += math.log(sentenceProb, 2)
	return '%.3f'%(prob)

# solving bigram add-one probability
def bigramAddOneProbability(sentence, unigramWordCount, bigramWordCount):
	lineAsList = sentence.split()
	prob = 1.0
	v = vocabularySize(unigramWordCount)
	for i in range(len(lineAsList) - 1):
		wordTuple = (lineAsList[i], lineAsList[i+1])
		if wordTuple not in bigramWordCount:
			prob *= (1.0 / (float(unigramWordCount[lineAsList[i]]) + v))
		else: 
			prob *= (bigramWordCount[wordTuple] + 1.0) / (float(unigramWordCount[lineAsList[i]]) + v)
	return prob

# problem 5 bigram add-one probability w/ parameters
def bigramAddOneProbability5(sentence, unigramWordCount, bigramWordCount):
	lineAsList = sentence.split()
	prob = 1.0
	v = vocabularySize(unigramWordCount)
	for i in range(len(lineAsList) - 1):
		wordTuple = (lineAsList[i], lineAsList[i+1])
		if wordTuple not in bigramWordCount:
			print(wordTuple,":  ",'%.3f'%(math.log((1.0 / (float(unigramWordCount[lineAsList[i]]) + v)),2)))
			prob *= (1.0 / (float(unigramWordCount[lineAsList[i]]) + v))
		else: 
			print(wordTuple,":  ",'%.3f'%(math.log((bigramWordCount[wordTuple] + 1.0) / (float(unigramWordCount[lineAsList[i]]) + v),2)))
			prob *= (bigramWordCount[wordTuple] + 1.0) / (float(unigramWordCount[lineAsList[i]]) + v)
			
	return prob


# solving bigram add-one log probability
def logBigramAddOneProbability(sentences, unigramWordCount, bigramWordCount):
	prob = 0.0
	for line in sentences.split("\n"):
		sentenceProb = bigramAddOneProbability(line, unigramWordCount, bigramWordCount)
		if sentenceProb != 0: 
			prob += math.log(sentenceProb, 2)
			
	return prob


# problem 5 log bigram probability w/ parameters
def logBigramAddOneProbability5(sentences, unigramWordCount, bigramWordCount):
	prob = 0.0
	for line in sentences.split("\n"):
		sentenceProb = bigramAddOneProbability5(line, unigramWordCount, bigramWordCount)
		if sentenceProb != 0: 
			prob += math.log(sentenceProb, 2)
	return '%.3f'%(prob)


#  Solving perplexity of a given sentence/corpus
def perplexity(model, sentences, unigramWordCount, bigramWordCount, allWordsTrainingCorpus):
	M = 0
	l=0

	for words in sentences.split():
		M += 1
	
	if model == "unigram":
		if logUnigramProbability(sentences, unigramWordCount, allWordsTrainingCorpus) == "undefined":
			return "undefined"
		else:
			l = (1.0/M) * logUnigramProbability(sentences, unigramWordCount, allWordsTrainingCorpus)
			print(l)
	elif model == "bigram":
		if logBigramProbability(sentences, unigramWordCount, bigramWordCount) == "undefined":
			return "undefined"
		else:
			l = (1.0/M) * logBigramProbability(sentences, unigramWordCount, bigramWordCount)
			print(l)
	else:
		if logBigramAddOneProbability(sentences, unigramWordCount, bigramWordCount) == "undefined":
			return "undefined"
		else:
			l = (1.0/M) * logBigramAddOneProbability(sentences, unigramWordCount, bigramWordCount)
			print(l)

	return '%.3f'%(math.pow(2,-1*l))


	
main()