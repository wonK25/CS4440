# -------------------------------------------------------------------------
# AUTHOR: your name
# FILENAME: title of the source file
# SPECIFICATION: description of the program
# FOR: CS 5990 (Advanced Data Mining) - Assignment #1
# TIME SPENT: how long it took you to complete the assignment
# -----------------------------------------------------------*/

# IMPORTANT NOTE: DO NOT USE ANY ADVANCED PYTHON LIBRARY TO COMPLETE THIS CODE SUCH AS numpy or pandas.
# You have to work here only with standard dictionaries, lists, and arrays

# Importing some Python libraries
import csv
from sklearn.metrics.pairwise import cosine_similarity

documents = []

# reading the documents in a csv file
with open('cleaned_documents.csv', 'r') as csvfile:
  reader = csv.reader(csvfile)
  for i, row in enumerate(reader):
      if i > 0: # skipping the header
          documents.append(row)

# Building the document-term matrix by using binary encoding.
# You must identify each distinct word in the collection using the white space as your character delimiter.
# --> add your Python code here

docTermMatrix = []
tokenized_docs = []
word_to_index = {}            #vocabulary dictionary
word_index = 0

# Tokenize documents and build the vocabulary
for doc in documents:
    # Assuming the text is in the second column (index 1)
    text = doc[1]
    words = text.split()                      # "I LOVE FOOD" to split -> ["I", "LOVE", "FOOD"]
    tokenized_docs.append(words)              # add the list of words into the tokenized_docs {"I":0, "LOVE":1, "FOOD":2}
    for word in words:                        # looping each word in the list to check if the word is alread in the vocab
        if word not in word_to_index:         # If we have "I LOVE FOOD AND LOVE DOG," here we have one more "LOVE" which does not increment index,
            word_to_index[word] = word_index  # but "DOG" gets index {"DOG":3}
            word_index += 1                   # word_index increments.

# Create the binary encoded document-term matrix
vocab_size = len(word_to_index)               # count words e.g 3
for  doc in tokenized_docs:                   # e.g doc = ["LOVE", "DOG"], tokenized_docs = [["LOVE", "DOG"], ["LOVE", "CAT"]]
# for i, doc in enumerate(tokenized_docs):
    vector = [0] * vocab_size                 # Create a vector of zeros for each document e.g. vector = [0, 0, 0]
    for word in doc:                          # Get the index of the word from the vocabulary {"LOVE": 0, "DOG": 1, "CAT": 2}
        index = word_to_index[word]           # "LOVE" is in the doc,
        vector[index] = 1                     # Set the value("LOVE") at that index to 1 (binary encoding), so vector = [1,0,0]
    docTermMatrix.append(vector)              # [["LOVE", "DOG"], ["LOVE", "CAT"]] -> [[1,1,0], [1,0,1]]
    # if i == 0:
    #    break
#print(docTermMatrix)

# Compare the pairwise cosine similarities and store the highest one
# Use cosine_similarity([X], [Y]) to calculate the similarities between 2 vectors
# --> Add your Python code here

# Initialize variables 
doc1, doc2, mostSimilar = -1, -1, -1.0                                          # The lowest similar value cosin180 degree = -1
n = len(docTermMatrix)                                                          # Get the total number of documents.

# Loop through each document by its index 'i' to select the first document of a pair.
for i in range(n):                                                              # check first vector
    #if i % 100 == 0:
        #print(f"similarity row {i}/{n}")
    for j in range(i + 1, n):                                                   # check next vector
        sim = cosine_similarity([docTermMatrix[i]], [docTermMatrix[j]])[0][0]   # Calculate the cosine similarity between the vector for doc i and doc j.
        if sim > mostSimilar:                                                   # Check if the similarity of the current pair is the highest one found so far.
            mostSimilar, doc1, doc2 = sim, i, j                                 # If it's a new record, update the highest score and store the indices of this pair.


# Print the highest cosine similarity following the information below
# The most similar documents are document 10 and document 100 with cosine similarity = x
# --> Add your Python code here

# Adding 1 (e.g., "document 1" instead of "document 0")
print(f"The most similar documents are document {doc1 + 1} and document {doc2 + 1} with cosine similarity = {mostSimilar:.4f}")