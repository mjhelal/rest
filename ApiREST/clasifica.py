#!/usr/bin/env python
# coding: utf-8

# # REST Api que clasifica reclamos ALFA

# In[1]:

print('comienzo de programa')

# Imports para uso general
import numpy as np
import pandas as pd
#import scipy.sparse
import time
import json
from sklearn.metrics import accuracy_score
import pickle

# Import de procesador de texto NLP
import HmNLP

# Imports de Api REST
from flask import Flask, request, redirect, url_for, flash, jsonify
from flask_restful import reqparse, abort, Api, Resource

#Import para persistir en base MongoDB
from pymongo import MongoClient

# # 1 Levanto el modelo y el vectorizer desde disco

# Levanto el modelo
from sklearn import model_selection
# load the model from disk
SVM = pickle.load(open('fiveClassSVC_model.sav', 'rb'))

# Levanto el vectorizer
f = open("Tfidf_vect.pkl", "rb")
Tfidf_vect = pickle.load(f)

# levanto tabla para decoding de grupos
tablaPerform = pd.read_csv("TablaPerformance.csv")
tablaPerform.set_index('Unnamed: 0', inplace=True)

#Me conecto a la base MongoDB 
client = MongoClient("mongodb://bigdatateam4:816D4747E4M@hdp-dmz-app1.in.iantel.com.uy/casosAlfa")
apiLog = client.casosAlfa['apiLog'] 

# # 2 Definir la api en Flask

app = Flask(__name__)
api = Api(app)


@app.route('/')
def index():
    return 'Server Works!'


class PredictDerivation(Resource):
    def post(self):
        # Obtenemos los datos, concateno descripcion y resumen ya que así los espera el modelo
        data = request.get_json(force=True)
        idCaso = data["id"]
        texto = data["Descripcion"] + ' ' + data["Resumen"]
                       
        # al vectorizador hay que pasarle una lista de strings (cada string tiene a su vez una lista con las palabras)
        TextoListo = []
        # paso el texto por el NLP processing
        TextoListo.append(str(HmNLP.procesoNLP(texto)))
        # Aplico el TFIDF Vectorizing
        TextoCaso = Tfidf_vect.transform(TextoListo)

        # Llamo sl clasificador para que me devuelva la categoria
        Clasification_SVM = SVM.predict(TextoCaso)
        confianza = 0
        confianza = tablaPerform['precision'][Clasification_SVM[0]]
        GrupoSugerido = tablaPerform['Grupo'][Clasification_SVM[0]]
        if GrupoSugerido != 'otro':
            output = {'id': idCaso, 'Clasificacion': GrupoSugerido, 'Confianza': str(round(confianza, 2))}
        else:
            output = {'id': idCaso, 'Clasificacion': 'Sin clasificar'}
        
        #grabo en la base MongoDB el request y el respose
        logRecord = {'request': data, 'response':output}
        apiLog.insert_one(logRecord)
        
        return jsonify(output)
        
        

# In[6]:


api.add_resource(PredictDerivation, '/clasi')
app.run(debug=True, host='0.0.0.0', port='5001')

print('fin de programa')
