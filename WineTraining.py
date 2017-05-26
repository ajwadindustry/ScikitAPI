import traceback
import time
import shutil
import os
import sys
from flask import Flask, request, jsonify
import pandas as pd
from sklearn.externals import joblib
from sklearn.neighbors import KNeighborsClassifier


app=Flask(__name__)

CSVFilename='C:\\Users\\Ajwad\\Downloads\\winequality-red.csv'
trainingdata=""
Variables=["citric acid","density","pH","alcohol","quality"]
dependent_variable = Variables[-1]
model_directory = 'model'
model_file_name = '%s/model.pkl' % model_directory
model_columns_file_name = '%s/model_columns.pkl' % model_directory

# These will be populated at training time
model_columns = None
Classifier = None



@app.route('/predict', methods=['POST'])
def predict():
    if Classifier:
        try:
            PostValue=request.json
            query=pd.DataFrame(PostValue)
            prediction=Classifier.predict(query)

            return jsonify({'prediction': prediction})

        except Exception as e:

            return jsonify({'error': str(e), 'trace': traceback.format_exc()})
    else:
        print('train first')
        return 'no model her'

@app.route('/train', methods=['GET'])
def train():

    csvDataSet=pd.read_csv(CSVFilename,delimiter=';')
    Dataset=csvDataSet[Variables]

    X=Dataset[Dataset.columns.difference([dependent_variable])]
    Y=Dataset[dependent_variable]

    global model_columns
    model_columns = list(X.columns)
    joblib.dump(model_columns, model_columns_file_name)

    global Classifier
    Classifier=KNeighborsClassifier(n_neighbors=10)
    start = time.time()
    Classifier.fit(X,Y)
    print('Trained in %.1f seconds' % (time.time() - start))
    print('Model training score: %s' % Classifier.score(X,Y))

    joblib.dump(Classifier, model_file_name)

    return 'Success'




@app.route('/wipe', methods=['GET'])
def wipe():
    try:
        shutil.rmtree('model')
        os.makedirs(model_directory)
        return 'Model wiped'

    except Exception as e:
        print (str(e))
        return 'Could not remove and recreate the model directory'


if __name__ == '__main__':
    try:
        port = int(sys.argv[1])
    except Exception as e:
        port = 80

    try:
        Classifier = joblib.load(model_file_name)
        print('model loaded')
        model_columns = joblib.load(model_columns_file_name)
        print('model columns loaded')

    except Exception as e:
        print('No model here')
        print('Train first')
        print(str(e))
        Classifier = None

    app.run(port=port, debug=True)

"""

csvDataSet=pd.read_csv('C:\\Users\\Ajwad\\Downloads\\winequality-red.csv',delimiter=';')
Variables=["citric acid","density","pH","alcohol","quality"]
dependent_variable = Variables[-1]




Classifier.fit(X,Y)
print(Classifier.predict([0,0.997,3.2,9.5]))

#for elements in X:
 #   print(elements)

"""