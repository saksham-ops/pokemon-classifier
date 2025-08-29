from flask import Flask,request,render_template,send_file
from src.pipelines.PredictionPipeline import Prediction_pipeline,CustomData,Batch_prediction
from src.pipelines.TrainingPipeline import Training_pipeline
from src.exception import CustomException
import os,sys


application=Flask(__name__)

app=application

@app.route('/')
def home_page():
    return render_template('index.html')

@app.route('/train')
def train_data():
    try:
        train_pipeline = Training_pipeline()
        train_pipeline.run_training_pipeline()
        
        return render_template('index1.html')
    except Exception as e:
        raise CustomException(e,sys)
        
@app.route('/predict',methods=['GET','POST'])
def predict_datapoint():
    if request.method=='GET':
        return render_template('form.html')
    
    else:
        data=CustomData(
            Total=float(request.form.get('Total')),
            HP = float(request.form.get('HP')),
            Attack = float(request.form.get('Attack')),
            Defense = float(request.form.get('Defense')),
            Sp_Atk = float(request.form.get('Sp_Atk')),
            Sp_Def = float(request.form.get('Sp_Def')),
            Speed = float(request.form.get('Speed')),
            hasMegaEvolution = request.form.get('hasMegaEvolution'),
            Height_m = float(request.form.get('Height_m')),
            Weight_kg = float(request.form.get('Weight_kg')),
            Catch_Rate = float(request.form.get('Catch_Rate')),
        )
        final_new_data=data.get_data_as_dataframe()
        predict_pipeline = Prediction_pipeline()

        pred=predict_pipeline.initiate_prediction(final_new_data)

        results=pred

        return render_template('form.html',final_result=results)
    
@app.route('/predict_file',methods=['GET','POST'])

def predict_file():

    try:
        if request.method=="POST":
            
            prediction_obj = Batch_prediction(request)
            data = prediction_obj.save_data()
            Prediction_path = prediction_obj.initiate_file_prediction(data)

            return send_file(path_or_file=Prediction_path,download_name=Prediction_path,as_attachment=True)
        
        else:
            return render_template('uploadfile.html')

    except Exception as e:
        raise CustomException(e,sys)

if __name__=="__main__":
    app.run()

