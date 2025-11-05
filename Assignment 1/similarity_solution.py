# -------------------------------------------------------------------------
# AUTHOR: your name
# FILENAME: title of the source file
# SPECIFICATION: description of the program
# FOR: CS 4440 (Data Mining) - Assignment #1
# TIME SPENT: how long it took you to complete the assignment
# -----------------------------------------------------------*/

#IMPORTANT NOTE: DO NOT USE ANY ADVANCED PYTHON LIBRARY TO COMPLETE THIS CODE SUCH AS numpy or pandas.
#You have to work here only with standard dictionaries, lists, and arrays

# Importing some Python libraries
import csv
from sklearn.metrics.pairwise import cosine_similarity

documents = []

#reading the documents in a csv file
with open('cleaned_documents.csv', 'r') as csvfile:
  reader = csv.reader(csvfile)
  for i, row in enumerate(reader):
      if i > 0: #skipping the header
         documents.append (row[1])

#Building the document-term matrix by using binary encoding.
#You must identify each distinct word in the collection using the white space as your character delimiter.

vocabulary = set()
for doc in documents:
    words = doc.split()
    vocabulary.update(words)

docTermMatrix = []
for doc in documents:
    words = set(doc.split())
    row = []
    for word in vocabulary:
        if word in words:
            row.append(1)
        else:
            row.append(0)
    docTermMatrix.append(row)

# Compare the pairwise cosine similarities and store the highest one
# Use cosine_similarity([X], [Y]) to calculate the similarities between 2 vectors
# --> Add your Python code here

max_similarity = 0
most_similar_docs = (None, None)

for i in range(len(docTermMatrix)):
    for j in range(i + 1, len(docTermMatrix)):  # Avoid self-comparison
        print("Comparing " + str(i) + " with " + str(j))
        sim = cosine_similarity([docTermMatrix[i]], [docTermMatrix[j]])[0][0]
        if sim > max_similarity:
            max_similarity = sim
            most_similar_docs = (i + 1, j + 1)

# Print the highest cosine similarity following the information below
# The most similar documents are document 10 and document 100 with cosine similarity = x

print(f"The most similar documents are document {most_similar_docs[0]} and document {most_similar_docs[1]} "
      f"with cosine similarity = {max_similarity:.4f}")
