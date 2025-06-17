import pandas as pd
import pickle
from flask import Flask, redirect, url_for, request, render_template,jsonify
import sqlite3
from string import punctuation
from nltk.corpus import stopwords
import nltk
from nltk.stem import WordNetLemmatizer
import string
import re
import numpy as np
import os
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.model_selection import train_test_split
from tensorflow.keras.models import load_model



stop_words = set(stopwords.words('english'))
lemmatizer = WordNetLemmatizer()



app = Flask(__name__)

def cleanData(doc):
    tokens = doc.split()
    table = str.maketrans('', '', punctuation)
    tokens = [w.translate(table) for w in tokens]
    tokens = [word for word in tokens if word.isalpha()]
    tokens = [w for w in tokens if not w in stop_words]
    tokens = [word for word in tokens if len(word) > 1]
    tokens = [lemmatizer.lemmatize(token) for token in tokens]
    tokens = ' '.join(tokens)
    return tokens

dataset = pd.read_csv('Dataset/dataset.csv', encoding ="ISO-8859-1")
labels = dataset['Source'].unique().tolist()
symptoms = dataset.Target
diseases = dataset.Source
Y = []
for i in range(len(diseases)):
    index = labels.index(diseases[i])
    Y.append(index)

X = []

for i in range(len(symptoms)):
    arr = symptoms[i]
    arr = arr.strip().lower()
    arr = arr.replace("_", " ")
    X.append(cleanData(arr))

vectorizer = TfidfVectorizer(use_idf=True, smooth_idf=False, norm=None, decode_error='replace')
tfidf = vectorizer.fit_transform(X).toarray()
X = pd.DataFrame(tfidf, columns=vectorizer.get_feature_names())


Y = np.asarray(Y)

model_path2 = 'model.h5' 

classifier = load_model(model_path2)


def getDrugs(disease):
    drugs = {
        'fungal infection': ['Antifungal Cream', 'Fluconazole', 'Itraconazole'],
        'allergy': ['Antihistamines', 'Corticosteroids', 'Epinephrine'],
        'gerd': ['Proton Pump Inhibitors', 'H2 Blockers', 'Antacids'],
        'chronic cholestasis': ['Ursodeoxycholic Acid'],
        'drug reaction': ['Antihistamines', 'Corticosteroids', 'Epinephrine'],
        'peptic ulcer disease': ['Proton Pump Inhibitors', 'H2 Blockers', 'Antacids'],
        'aids': ['Antiretroviral Therapy (ART)'],
        'diabetes': ['Metformin', 'Insulin', 'Sitagliptin'],
        'gastroenteritis': ['Oral Rehydration Solution', 'Antibiotics (in bacterial gastroenteritis)'],
        'bronchial asthma': ['Bronchodilators', 'Inhaled Corticosteroids', 'Leukotriene Modifiers'],
        'hypertension': ['ACE Inhibitors', 'Beta-blockers', 'Calcium Channel Blockers'],
        'migraine': ['Triptans', 'NSAIDs', 'Beta-blockers'],
        'cervical spondylosis': ['Pain Relievers', 'Muscle Relaxants', 'Physical Therapy'],
        'paralysis (brain hemorrhage)': ['Rehabilitation', 'Physical Therapy', 'Occupational Therapy'],
        'jaundice': ['Ursodeoxycholic Acid', 'Vitamin K', 'Liver Supportive Medications'],
        'malaria': ['Chloroquine', 'Artemether', 'Lumefantrine'],
        'chicken pox': ['Acyclovir', 'Antihistamines', 'Calamine Lotion'],
        'dengue': ['Fluid Replacement Therapy', 'Pain Relievers', 'Monitoring'],
        'typhoid': ['Antibiotics (e.g., Ciprofloxacin, Ceftriaxone)'],
        'hepatitis a': ['Hepatitis A Vaccine', 'Supportive Care'],
        'hepatitis b': ['Hepatitis B Vaccine', 'Antiviral Medications (e.g., Tenofovir)'],
        'hepatitis c': ['Antiviral Medications (e.g., Sofosbuvir, Ledipasvir)'],
        'hepatitis d': ['Hepatitis D Vaccine', 'Antiviral Medications'],
        'hepatitis e': ['Supportive Care', 'Avoidance of Alcohol'],
        'alcoholic hepatitis': ['Corticosteroids', 'Pentoxifylline', 'Nutritional Support'],
        'tuberculosis': ['Isoniazid', 'Rifampin', 'Pyrazinamide', 'Ethambutol'],
        'common cold': ['Pain Relievers', 'Decongestants', 'Antihistamines'],
        'pneumonia': ['Antibiotics (e.g., Amoxicillin, Azithromycin, Levofloxacin)'],
        'dimorphic hemorrhoids (piles)': ['Pain Relievers', 'Stool Softeners', 'Dietary Changes'],
        'heart attack': ['Aspirin', 'Beta-blockers', 'ACE Inhibitors'],
        'varicose veins': ['Compression Stockings', 'Venotonics'],
        'hypothyroidism': ['Levothyroxine'],
        'hyperthyroidism': ['Beta-blockers', 'Antithyroid Drugs (e.g., Methimazole, Propylthiouracil)'],
        'hypoglycemia': ['Oral Glucose', 'Glucagon', 'Dextrose'],
        'osteoarthritis': ['Pain Relievers', 'Topical Treatments', 'Physical Therapy'],
        'arthritis': ['NSAIDs', 'DMARDs', 'Biologic Agents'],
        '(vertigo) paroxysmal positional vertigo': ['Epley Maneuver', 'Vestibular Rehabilitation'],
        'acne': ['Topical Treatments (e.g., Benzoyl Peroxide, Retinoids)', 'Oral Medications (e.g., Antibiotics, Isotretinoin)'],
        'urinary tract infection': ['Antibiotics (e.g., Trimethoprim-Sulfamethoxazole, Ciprofloxacin)'],
        'psoriasis': ['Topical Treatments (e.g., Corticosteroids, Vitamin D Analogues)', 'Systemic Treatments (e.g., Methotrexate, Biologic Agents)'],
        'impetigo': ['Topical Antibiotics (e.g., Mupirocin)', 'Oral Antibiotics (e.g., Cephalexin, Dicloxacillin)']
    }
    return drugs.get(disease, [])


def getDiet(filepath):
    diet = ""
    if os.path.exists("diets/"+filepath+".txt"):
        with open("diets/"+filepath+".txt", "r") as file:
            lines = file.readlines()
            for i in range(len(lines)):
                diet += lines[i]+"\n"
        file.close()
    else:
        with open("diets/others.txt", "r") as file:
            lines = file.readlines()
            for i in range(len(lines)):
                diet += lines[i]+"\n"
        file.close()
    return diet

 
@app.route("/")
def hello_world():
    return render_template("home.html")

@app.route('/logon')
def logon():
	return render_template('signup.html')

@app.route('/login')
def login():
	return render_template('signin.html')

@app.route('/home')
def home():
	return render_template('home.html')

@app.route('/index')
def index():
	return render_template('index.html')

@app.route("/signup")
def signup():

    username = request.args.get('user','')
    name = request.args.get('name','')
    email = request.args.get('email','')
    number = request.args.get('mobile','')
    password = request.args.get('password','')
    con = sqlite3.connect('signup.db')
    cur = con.cursor()
    cur.execute("insert into `info` (`user`,`email`, `password`,`mobile`,`name`) VALUES (?, ?, ?, ?, ?)",(username,email,password,number,name))
    con.commit()
    con.close()
    return render_template("signin.html")

@app.route("/signin")
def signin():

    mail1 = request.args.get('user','')
    password1 = request.args.get('password','')
    con = sqlite3.connect('signup.db')
    cur = con.cursor()
    cur.execute("select `user`, `password` from info where `user` = ? AND `password` = ?",(mail1,password1,))
    data = cur.fetchone()

    if data == None:
        return render_template("signin.html")    

    elif mail1 == 'admin' and password1 == 'admin':
        return render_template("index.html")

    elif mail1 == str(data[0]) and password1 == str(data[1]):
        return render_template("index.html")
    else:
        return render_template("signup.html")




@app.route('/predict', methods=['GET'])
def predict():
    if request.method == 'GET':
        question = request.args.get('mytext', False)
        question = question.strip("\n").strip()
        
        # Preprocess the input data
        arr = question
        arr = arr.strip().lower()
        arr = arr.replace("_", " ")
        testData = vectorizer.transform([cleanData(arr)]).toarray()
        
        # Make prediction
        temp = testData.reshape(testData.shape[0], testData.shape[1], 1, 1)
        predict = classifier.predict(temp)
        predict = np.argmax(predict)
        output = labels[predict]
        
        # Get diet recommendation
        diet = getDiet(output)
        
        # Get drugs for the predicted disease
        drugs = getDrugs(output)
        
        # Log the question and predicted output
        print(question + " " + output)
        
        # Perform additional actions here, such as saving data to a database or logging
        
        # Return the response
        return jsonify({"response": "Disease Predicted as " + output + "\n\n" + "Diet: " + diet + "\n\n" + "Drugs: " + ", ".join(drugs)})


@app.route('/note')
def note():
	return render_template('notebook.html')



if __name__ == '__main__':
    app.run(debug=False)