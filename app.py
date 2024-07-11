from flask import Flask, render_template, request, jsonify, redirect, url_for
import torch
import numpy as np
from utility import actions,preprocess
import bert_model_arch
from neo4j import GraphDatabase
import csv



#establish connection with neo4j database
with open("cred.txt") as file:
    data=csv.reader(file,delimiter=',')
    for row in data:
        username=row[0]
        password=row[1]
        uri=row[2]
driver=GraphDatabase.driver(uri=uri,auth=(username,password))
session=driver.session()

#integrating ML model and rendering basic front-end webpage to take input of instruction from user
app=Flask(__name__)

device = 'cuda' if torch.cuda.is_available() else 'cpu'
model=bert_model_arch.BERTmodel()
model.load_state_dict(torch.load('bert_model.pt',map_location=device))
model.eval()

@app.route('/',methods=['GET'])
def landing():
    return render_template ('index.html')

@app.route('/',methods=['POST'])
def predict():
    if request.method=='POST':
        sentence = request.form.get('sentence')
        if not sentence:
            return "No sentence found",400
        processed=preprocess(sentence)
        prediction=model(processed['input_ids'],processed['attention_mask'])
        prediction=np.argmax(prediction.detach().cpu().numpy(),axis=1)[0]
        
        if prediction==0:
            return redirect('/addNode')
        elif prediction==1:
            return redirect('/deleteNode')
        # elif prediction==2:
        #     #add property
        # elif prediction==3:
        #     #add relationship

        action=actions[prediction]
    # return render_template('index.html',action=action)    
    

'''end-points to manipulate graph based on prediction made by the model'''

#add node
@app.route('/addNode',methods=['GET','POST'])
def addNode():
    if request.method=='POST':
        label = request.form.get('label')
        query = f"CREATE (n:{label})"
        param={"label":label} #is to map param values to placeholder variables
        try:
            session.run(query,param)
            return render_template('commonMessage.html',action="add node")
        except Exception as e:
            return (str(e))
    return render_template('addNode.html')   

#delete node
@app.route('/deleteNode',methods=['GET','POST'])
def deleteNode():
    if request.method == 'POST':
        delete_option = request.form.get('deleteOption')
        label = request.form.get('label')
        prop = request.form.get('prop')

        query = ""
        if delete_option == 'label' and label:
            query = f"MATCH (n:{label}) DELETE n"
        elif delete_option == 'property' and prop:
            query = f"MATCH (n) WHERE n.prop = '{prop}' DELETE n"

            try:
                session.run(query)
                return render_template('commonMessage.html',action="delete node")
            except Exception as e:
                return str(e)
    return render_template('deleteNode.html')

#add property

#add relationship

if __name__=='__main__':
    app.run(port=3000,debug=True)  