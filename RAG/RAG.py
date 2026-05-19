import os
import pandas as pd
import numpy as np
import google.generativeai as genai
from collections import Counter
from sentence_transformers import SentenceTransformer
from sklearn.metrics.pairwise import cosine_similarity
import time
import nltk
from nltk.corpus import stopwords
from dotenv import load_dotenv

# Download NLTK data if not already present
try:
    nltk.data.find('tokenizers/punkt')
    nltk.data.find('corpora/stopwords')
except LookupError:
    nltk.download('punkt')
    nltk.download('stopwords')

# Load environment variables
load_dotenv()

# Loading and Init
# API KEY Load the API
API_KEY = os.getenv('API_KEY')
if not API_KEY:
    API_KEY = input("Please enter your Google API Key: ").strip()
    
genai.configure(api_key=API_KEY)
print("API loaded successfully!")

# Load the model
model = genai.GenerativeModel("models/gemini-1.5-flash")   #   gemini-1.5-flash, 2.5-pro

# Load data
try:
    dataframe = pd.read_csv("data/data_scientist_salaries.csv")
    print("Loaded data from data/data_scientist_salaries.csv")
except FileNotFoundError:
    try:
        dataframe = pd.read_csv("data_scientist_salaries.csv")
        print("Loaded data from current directory")
    except FileNotFoundError:
        print("ERROR: Could not find data_scientist_salaries.csv")
        print("Please place the file in the 'data' folder or current directory")
        exit(1)

# we are storing the imp columns required for our data preparation part. impColumns are necessary to answer the questions asked to the RAG.
# our data set has too much data so we need to filter out important data
impCols = [ "Hobby", "OpenSource", "Country", "Student", "Employment", "FormalEducation", "UndergradMajor", "CompanySize", "DevType", "YearsCoding", "Salary", "SalaryType", "ConvertedSalary"]
embeddingModel = SentenceTransformer('all-MiniLM-L6-v2')

#Understand the data
# clean this dataset and handle null
def preprocessing(dataframe, impCols):

  # extract the data from the dataset only of impCols initialised above
  extractedData = dataframe[impCols].copy()

  #### Handling null or empty values
  for col in impCols:

    # identify if the extracted data is null or not and count of missing values > 0
    if extractedData[col].isnull().sum() > 0:
      # Missing values are present
      # print(col)

      # for missing data in text type colums datatype = object we replace it with "unknown"
      if extractedData[col].dtype == 'object':
        # if object is null add replace the cell with "unknown"
        extractedData[col] = extractedData[col].fillna('unknown')
        # converting it into lower case - ONLY for text columns
        extractedData[col] = extractedData[col].astype(str).str.lower()
        # extractedData[col] = str(extractedData[col]).lower()

      # for missing data in number type columns
      elif col == "ConvertedSalary":
        # if any of the cells from ConvertedSalary column are missing then add the median as the default value
        extractedData[col] = extractedData[col].fillna(extractedData[col].median())

      # For any other columns empty values will be replaced by 0
      else:
        extractedData[col] = extractedData[col].fillna(0)

  if 'ConvertedSalary' in extractedData.columns:
    # converting values to number "50" -> 50 and if there is a value that cant be converted to a number then put NaN.
    extractedData['ConvertedSalary'] = pd.to_numeric(extractedData['ConvertedSalary'], errors='coerce')
    extractedData['ConvertedSalary'] = extractedData['ConvertedSalary'].fillna(extractedData['ConvertedSalary'].median())

  #### Generalising the years coding data to handle all the data range values
  if 'YearsCoding' in extractedData.columns:

    # group years of experience. this is useful for any math related operations or even answering que related to ranges like below 5years
    # midpoint values will be used. bcz this will give us equal spacing between the levels
    yearsexp = {
        "0-2 years":1, "3-5 years": 4, "6-8 years": 7, "9-11 years": 10, "12-14 years": 13, "15-17 years": 16, "18-20 years": 19,
        "21-23 years":22, "24-26 years": 25, "27-29 years": 28, "30 or more years":30
    }
    # take YearsCoding column, convert it to corresponding num value and then map it to the corresp num value in the yearsexp dict
    extractedData['YearsCodingNum'] = extractedData['YearsCoding'].map(yearsexp)

    # if there are any values like null replace those with median value of YearsCodingNum like if the other vals were 1, 4, 7 then fill null values with median (1,4,7)
    extractedData['YearsCodingNum'] = extractedData['YearsCodingNum'].fillna(extractedData['YearsCodingNum'].median())

  # this extractedData is our cleaned version of original dataframe
  return extractedData

# Compute Embeddings
# We are computing embeddings for all the cells and rows present in the csv file. This is done so that after a question is asked by the user we can retrieve the
# answer quickly. A text representation can be done which can be used later for semantic search and retrieval of answers ()

def getEmbeddingsofCSV(extractedData):
  # 1. Convert csv to text
  completeText = []
  for _, rowData in extractedData.iterrows():
    # we are converting table data to text so that we can search on this text later.
    retrievedText = []
    if 'ConvertedSalary' in rowData:
      if rowData['ConvertedSalary'] > 0:
        retrievedText.append(f"Converted Salary -> "+str(rowData['ConvertedSalary']))
    if 'YearsCoding' in rowData:
      # if rowData['YearsCoding'] is not None and str(rowData['YearsCoding']).strip():
      if str(rowData['YearsCoding']).strip():
        retrievedText.append(f"Years Coding -> "+str(rowData['YearsCoding']))
        # print("years coding retrieved rows = "+str(len(rowData['YearsCoding'])))
        if 'YearsCodingNum' in rowData:
          retrievedText.append(f"Years Coding Num -> "+str(rowData['YearsCodingNum']))
    if 'Country' in rowData:
      if rowData['Country'] is not None:
        retrievedText.append(f"Country -> "+str(rowData['Country']))
    if 'CompanySize' in rowData:
      # if rowData['CompanySize'] is not None and str(rowData['CompanySize']).strip():
      if str(rowData['CompanySize']).strip():
        retrievedText.append(f"Company Size -> "+str(rowData['CompanySize']))
    if 'DevType' in rowData:
      if rowData['DevType'] is not None:
        retrievedText.append(f"DevType -> "+str(rowData['DevType']))
    if 'FormalEducation' in rowData:
      if rowData['FormalEducation'] is not None:
        retrievedText.append(f"FormalEducation -> "+str(rowData['FormalEducation']))

    completeText.append("; ".join(retrievedText))

  # Compute the embeddings for semantic search for the complete Text we got from csv
  textembeddingsFromCSV = embeddingModel.encode(completeText)
  return completeText, textembeddingsFromCSV

#####Matching functions or Retriever Functions

## 1. Keyword Matching
def keywordMatching(userQuery, textRep, extractedData):

  eng_stopwords = set(stopwords.words('english'))
  queryWords = []

  # first we extract every unique word from the userQuery and only words which are not stopwords
  for word in userQuery.lower().split():
    clean_word = word.strip(",.!?:;()[]{}")
    # check if word is not a stopword AND has meaningful length (>2 characters)
    if clean_word not in eng_stopwords and len(clean_word) > 2:
      queryWords.append(clean_word)

  # FALLBACK: If all words were stopwords, use original words (but still filter very short ones)
  if not queryWords:
    # Create fallback list with words longer than 2 characters
    fallback_words = []
    for word in userQuery.lower().split():
      clean_word = word.strip(",.!?:;()[]{}")
      if len(clean_word) > 2:
        fallback_words.append(clean_word)
    queryWords = fallback_words

  keywordMatchScores = []

  # compare words in textRep = ["Role: Data Scientist; Experience: 5-10 years; ...] and in the query get a score
  for index, rowData in enumerate(textRep):
    scoreObtained = 0      # number of matching terms or words found in a row
    for word in queryWords:
        # word should be in a row of CSV then only increase score of that row
        # Use word in rowData.lower() for matching
        if word in rowData.lower():
          scoreObtained += 1

    # Only add rows that have at least one match (score > 0)
    if scoreObtained > 0:
      keywordMatchScores.append((scoreObtained, index))   # score= no of matching words and index = row number

  # sort with highest matching row first
  keywordMatchScores.sort(reverse = True)

  temp = []
  for kmscore, index in keywordMatchScores:
    if kmscore > 0:
      temp.append(index)
  indexesFinal = temp

  # get indexes of rows with score match above 0
  if indexesFinal:
    return extractedData.iloc[indexesFinal].copy()    # this is returning the actual data from the CSV that is present in those indexes
  else:
    return extractedData.head(0)

## 2. Semantic Matching
def semanticMatching(userQuery, EmbeddingsofCSV, extractedData):
  # convert the text into vector embedding. Embedding contains the semantic meaning and will help in matching
  embeddingofQuery = embeddingModel.encode([userQuery])
  similarityScoreValues = cosine_similarity(embeddingofQuery, EmbeddingsofCSV)[0]   # cosine similaroty would compare both and compare how similar both the vectors are. Same = 1, diff=-1
  indexesFinal = np.argsort(similarityScoreValues)[::-1][:5]    # returning indicies of highest similarity scores. only 5 rows
  return extractedData.iloc[indexesFinal].copy()


# ans Questions
def retriever(userQuestion, method, textRep, extractedData, EmbeddingsofCSV):

  # Implementing the question answering part to test the RAG
  # print("Query = "+str(userQuestion))
  # method could be semantic, keyword matching or hybrid
  # print("Method of ans retrieval = "+str(method))

  if method.lower() == 'keyword':
    dataObtained = keywordMatching(userQuestion, textRep, extractedData)

  elif method.lower() == 'semantic':
    dataObtained = semanticMatching(userQuestion, EmbeddingsofCSV, extractedData)
  
  else:
    print(f"Unknown method: {method}. Using semantic as default.")
    dataObtained = semanticMatching(userQuestion, EmbeddingsofCSV, extractedData)

  print("len of dataObtained retrieved data count = "+str(len(dataObtained)))
  contextList = []
  resFinal = []
  
  # take the dataObtained using those search methods and format that data to be able to use it as context.
  # loop through each row in dataObtained
  for i, (_, rowVal) in enumerate(dataObtained.iterrows()):

    contextList.append(f"\nEntry {i+1}:")

    # getting the context
    if rowVal['DevType'] != 'unknown':
        contextList.append(f"  Persons Role: {rowVal['DevType']}")

    if rowVal['YearsCoding'] != 'unknown':
        contextList.append(f"  YearsCoding: {rowVal['YearsCoding']}")

    if rowVal['Country'] != 'unknown':
        contextList.append(f"  Country: {rowVal['Country']}")

    if rowVal['ConvertedSalary'] > 0:
        contextList.append(f"  Salary: ${rowVal['ConvertedSalary']:,.0f}")

    # if rowVal['CompanySize']:
    if rowVal['CompanySize'] != 'unknown':
        contextList.append(f"  Company Size: {rowVal['CompanySize']}")

    # if rowVal['FormalEducation'] :
    if rowVal['FormalEducation'] != 'unknown':
        contextList.append(f"  Education: {rowVal['FormalEducation']}")

    # if rowVal['Employment']:
    if rowVal['Employment'] != 'unknown':
      contextList.append(f"  Employment: {rowVal['Employment']}")

  # get the full context as a string
  context = "\n".join(contextList)

  #### Creating a prompt that is clear and understandable
  prompt = "Hello. You are a data science analyst. There is some csv data given to you.\n"
  prompt += "You have to analyse the data and correctly answer the questions asked 'only' from the data given to you.\n"
  prompt += "For salary related questions analyse the data calculate averages and ranges.\n"
  prompt += "Here is the data context:\n"
  prompt += context + "\n\n"
  prompt += "The question is: "
  prompt += userQuestion

  # get the response for the given prompt
  try:
    response = model.generate_content(prompt)
    response_text = response.text
  except Exception as e:
    if "429" in str(e):
      response_text = "API quota exceeded. Please try again later or check your API key limits."
    else:
      response_text = f"Error: {str(e)}"

  res = {
      'question': userQuestion,
      'dataretrievalMethod': method,
      'dataObtained': dataObtained,
      'response': response_text
  }
  resFinal.append(res)
  return resFinal

if __name__ == "__main__":
    print("=" * 50)
    print("RAG Salary Assistant - Command Line Interface")
    print("=" * 50)
    
    # Test the system
    print("\nInitializing RAG system...")
    extractedData = preprocessing(dataframe, impCols)
    textRep, EmbeddingsofCSV = getEmbeddingsofCSV(extractedData)
    
    print(f"Total Rows = {len(extractedData)}")
    print(f"Total Columns = {len(extractedData.columns)}")
    print(f"Text representations created: {len(textRep)}")
    
    if 'YearsCoding' in extractedData.columns:
        print(f"30 or more years experience: {(extractedData['YearsCoding'] == '30 or more years').sum()}")
    
    print("\n" + "=" * 50)
    print("System ready! Enter your queries.")
    print("Format: question; method (keyword or semantic)")
    print("Example: What is the average salary for data scientists?; semantic")
    print("Type 'Exit' to quit")
    print("=" * 50 + "\n")
    
    while True:
        userInput = input("Enter query: ")
        
        if userInput.lower() == "exit":
            print("Goodbye!")
            break
        
        elif ";" not in userInput:
            print("\n Error: Invalid format. Please use 'question; method'")
            print("Example: What is the average salary?; semantic\n")
            
        else:
            inputs = userInput.split(';')
            userQues = inputs[0].strip()
            methodofRetrieval = inputs[1].strip().lower()
            
            if methodofRetrieval not in ['keyword', 'semantic']:
                print("\n Error: Method must be either 'keyword' or 'semantic'\n")
                continue
            
            print(f"\n🔍 Processing...")
            print(f"Question: {userQues}")
            print(f"Method: {methodofRetrieval}")
            
            responseObtained = retriever(userQues, methodofRetrieval, textRep, extractedData, EmbeddingsofCSV)
            
            for res in responseObtained:
                print("\n" + "=" * 50)
                print(" QUESTION:")
                print(res['question'])
                print("\n METHOD:")
                print(res['dataretrievalMethod'])
                print("\n ANSWER:")
                print(res['response'])
                print("=" * 50 + "\n")
