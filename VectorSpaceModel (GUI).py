#!/usr/bin/env python
# coding: utf-8

# In[182]:


#Vector Space Model 


# In[183]:


#Approach for Making Vector Space Model 
#1.Preprocessing -Removal of stopwords, lemmatization, tokenization
#2.TF-IDF Score 
#3.Query Processing.
#4. Cosine Similarity
#5. Ranking of Documents


# In[184]:


#Libraries 
import nltk #Lemmatization
from nltk.stem import WordNetLemmatizer
import numpy as np
import sklearn 
from sklearn.metrics.pairwise import cosine_similarity 

# In[185]:


lemmatizer = WordNetLemmatizer()



import tkinter as tk 
from PIL import Image, ImageTk
root = tk.Tk(className='Vector Space Model')
root.geometry('1250x500')
root.configure(bg="black")

background = ImageTk.PhotoImage(file="img1.jpg")
label1 = tk.Label(root,image=background)
label1.pack(fill=tk.BOTH,expand=True)
label = tk.Label(root,text="  Vector Space Model",bg="black",
fg="white",font="Courier 30 bold")
label.place(relx=0.25,rely=0.25,height=50)

# In[186]:


#Preprocessing
# Step1. Importing all the stopwords from the file for the removal. 

with open('Stopword-List.txt','r') as lines:
    stopwords = lines.readlines()
    stopwords = [x.rstrip() for x in stopwords]
    #print(stopwords)
    
# with-open automatically closes the file. 
# stop words are read and then stripped of, stored in stopwords.
    


# In[187]:


#Step2. Tokenization - Removal of Stopwords 
documents = {}
docs = []
for x in range(1,51):
    myfile=open("ShortStories/"+str(x)+ ".txt","r",encoding='utf-8')
    #for x in tokens:
    #print(x)
    
    words=myfile.read().replace(".","").replace("n't"," not").replace("'","").replace("]"," ").replace("[","").replace(","," ").replace("?","").replace("\n"," ").replace("-"," ").replace("(","").replace(")","").replace("!","").replace("“","").replace("”","").replace(":","").replace("*","").replace("—","").replace(";","").replace("’","").split()
    
    #for i in range(len(temp)):
    #temp[i]=[p_stemmer.stem(x.lower()) for x in temp[i]]
    lema=[lemmatizer.lemmatize(x.lower()) for x in words]
    tokens = [x for x in words if x not in stopwords]
    docs.append(tokens)
    documents[x]=tokens
    
    
myfile.close()

#len(documents)
#len(documents[3])


# In[188]:


#Combining the entire dataset into a single corpus.
corpus = []
def makecorpus(docs):
    for i in docs:
        if type(i) == list:
            makecorpus(i)
        else:
            corpus.append(i)
            
makecorpus(docs)


# In[189]:


len(corpus)


# In[190]:


#Unique identification 
corpus = list(set(corpus))
len(corpus)


# In[191]:


documentVectors={}

for x in range(1,51):
    documentVectors[x]=dict.fromkeys(corpus,0) 


# In[192]:


for x in range(1,51):
    for word in documents[x]:
        documentVectors[x][word]+=1   


# In[193]:


#Calculation of term-frequency --tf--
tf = {}
for x in range(1,51):
    tf[x]={}
    for word,count in documentVectors[x].items():
        tf[x][word]=count


# In[194]:


tf[23]['across']


# In[195]:


#Document wise unique tokens.
for x in range(1,50):
    documents[x]=set(documents[x])
    documents[x]=list(set(documents[x]))


# In[196]:


WordDocCount=dict.fromkeys(corpus,0)
for word in corpus:
    for x in range(1,51):
        if word in documents[x]:
            WordDocCount[word]+=1
        


# In[197]:


#WordDocCount


# In[198]:


#Calculation inverse document frequency --idf-- 
import math
idf = {}
for word in corpus:
    if WordDocCount[word] > 0:
        df=WordDocCount[word]
        if df > 50:
            df=50
    idf[word]=math.log(df/50)


# In[199]:


#Calculating tf-idf score
tfidf={}
for x in range(1,50):
    tfidf[x]={}
    for word in documentVectors[x]:
        tfidf[x][word]=tf[x][word]*idf[word]


# In[ ]:





# In[200]:



# Query Processing 

#Preprocessing of the Query. 
def inputval():
    query=query_.get()
    query=query.replace(".","").replace("n't"," not").replace("'","").replace("]"," ").replace("[","").replace(","," ").replace("?","").replace("\n"," ").replace("-"," ").replace("(","").replace(")","").replace("!","").replace("“","").replace("”","").replace(":","").replace("/"," / ").replace("*","").replace("—","").replace(";","").replace("’","").split()
    x=[lemmatizer.lemmatize(x.lower()) for x in query]
    print(x)

    queryVector = dict.fromkeys(corpus,0)
    for word in x:
        queryVector[word]+=1  


    for words in queryVector:
        queryVector[words]=queryVector[words]*idf[word]

    # Cosine Similarityy
    
    cosine = {}
    temp=0
    vector1=np.array([list(queryVector.values())])
    for x in range(1,50):
        vector2=np.array([list(tfidf[x].values())])
        if cosine_similarity(vector1,vector2)>0.005: #alpha value.
            temp=cosine_similarity(vector1,vector2)[0][0]
            cosine[x]=temp


    #sort_orders = sorted(cosine.items(), key=lambda x: x[1], reverse=True)
    label10 = tk.Label(root,text=f"Ret.Docs {cosine.keys()}",bg="black",fg="white",font="Courier 9 bold")
    label10.place(relx=0.0,rely=1.0,height=35,anchor='sw')



query_=tk.StringVar()

entry = tk.Entry(root,width=100,bd=3,textvariable=query_)
entry.place(relx=0.17,rely=0.45,height=35)

button = tk.Button(root,text="Search",bg="#000000",fg="white",
font="Courier 15 bold",command=inputval)
button.place(relx=0.73,rely=0.45,height=35)


root.mainloop()
# In[ ]:





# In[ ]:




