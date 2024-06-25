
import pandas as pd
import numpy as np
from sklearn.utils import shuffle

# skin 
import pathlib
from tensorflow.keras.preprocessing import image_dataset_from_directory

### Visualizing the training data
from tensorflow import keras
import numpy as np
from tensorflow.keras.preprocessing import image
import requests
from PIL import Image
from io import BytesIO

# **Read and shuffle the dataset**
df = pd.read_csv('dataset.csv')
df = shuffle(df,random_state=42)
df.head()




cols = df.columns
data = df[cols].values.flatten()

s = pd.Series(data)
s = s.str.strip()
s = s.values.reshape(df.shape)

df = pd.DataFrame(s, columns=df.columns)
df.head()



df = df.fillna(0)
df.head()








df1 = pd.read_csv('Symptom-severity.csv')
df1['Symptom'] = df1['Symptom'].str.replace('_',' ')
df1.head()





discrp = pd.read_csv("symptom_Description.csv")
discrp_ar = pd.read_csv("ar_description.csv")
discrp.head()




discrp_ar.head()


ektra7at = pd.read_csv("symptom_precaution.csv")
ektra7at_ar = pd.read_csv("ar_precaution.csv")
ektra7at.head()



ektra7at_ar.head()



Diseases = pd.read_csv("ar_Disease.csv")
Diseases.head()



symptoms = pd.read_csv("ar_Symptom.csv")
symptoms.head()



def predd(psymptoms,model):
    #print(psymptoms)
    a = np.array(df1["Symptom"])
    b = np.array(df1["weight"])

    for j in range(len(psymptoms)):
        for k in range(len(a)):
            if psymptoms[j]==a[k]:
                psymptoms[j]=b[k]

    psy = [psymptoms]
    pred2 = model.predict(psy)

    
    # get name of Disease in arabic.
    pred_name_ar = (Diseases[Diseases['Disease'] == pred2[0]])['Disease_Ar'].iloc[0]

    # get description for Disease in en and ar.
    disp= discrp[discrp['Disease']==pred2[0]]
    disp = disp.values[0][1]

    disp_ar= discrp_ar[discrp_ar['Disease']==pred2[0]]
    disp_ar = disp_ar.values[0][1]
    # print(disp_ar)
    # print('\n')
    
    # get precaution for Disease in en and ar
    # en
    c_en=np.where(ektra7at['Disease']==pred2[0])[0][0]
    precuation_list_en=[]
    for i in range(1,len(ektra7at.iloc[c_en])):
          precuation_list_en.append(ektra7at.iloc[c_en,i])

    
    # ar
    c_ar=np.where(ektra7at_ar['Disease']==pred2[0])[0][0]
    precuation_list_ar=[]
    for j in range(1,len(ektra7at_ar.iloc[c_ar])):
          precuation_list_ar.append(ektra7at_ar.iloc[c_ar,j])

    return pred2[0],pred_name_ar,disp,disp_ar,precuation_list_en,precuation_list_ar












def run_all_models(psymptoms, models):
    predictions = {}
    for model_name, model in models.items():
        disease, disease_name_ar, disp_en, disp_ar, precautions_list_en, precautions_list_ar = predd(psymptoms, model)
        if disease not in predictions:
            predictions[disease] = {
                "count": 0,
                "details": (disease_name_ar, disp_en, disp_ar, precautions_list_en, precautions_list_ar)
            }
        predictions[disease]["count"] += 1
    
    total_predictions = sum([pred["count"] for pred in predictions.values()])
    results = []
    
    for disease, data in predictions.items():
        confidence = (data["count"] / total_predictions) * 100
        disease_name_ar, disp_en, disp_ar, precautions_list_en, precautions_list_ar = data["details"]
        results.append({
            "disease": disease,
            "disease_name_ar": disease_name_ar,
            "confidence": confidence,
            "description_en": disp_en,
            "description_ar": disp_ar,
            "precautions_en": precautions_list_en,
            "precautions_ar": precautions_list_ar
        })
    
    return results


# ///////////////////////////////////////////////////////////////
# //////////////////////////////////////////////////////////////
# //////////////////////////////////////////////////////////////


class_names = ['actinic keratosis',
 'basal cell carcinoma',
 'dermatofibroma',
 'melanoma',
 'nevus',
 'pigmented benign keratosis',
 'seborrheic keratosis',
 'squamous cell carcinoma',
 'vascular lesion']


skin_disease_names_disc = pd.read_csv('skin-for-flask/skin_cancer_diseases.csv')
skin_disease_names_disc



# Function to get disease details by English name
def get_disease_details(disease_name):
    row = skin_disease_names_disc[skin_disease_names_disc['en_disease_name'] == disease_name]
    if not row.empty:
        return {
            'en_disease_name': row['en_disease_name'].values[0],
            'ar_disease_name': row['ar_disease_name'].values[0],
            'en_description': row['en_description'].values[0],
            'ar_description': row['ar_description'].values[0]
        }
    else:
        return None
    






# Load the model
skin_model = keras.models.load_model('skin-for-flask/Cancer.h5')








# # Flask api


from flask import Flask, request, jsonify,render_template
import pickle

app = Flask(__name__)


rnd_forest = pickle.load(open('Random_Forest.pkl','rb'))
tree = pickle.load(open('Decision_Tree.pkl','rb'))
nb_model = pickle.load(open('Gaussian_Naive_Bayes.pkl','rb'))
svm_model = pickle.load(open('Support_Vector_Machine.pkl','rb'))


 
@app.route("/")
def Home():
    return render_template('index.html')



@app.route("/predict",methods=["POST"])
def predictDisease():   
    data = request.get_json()
    psymptoms = []
    # print(data)

    for item in data:
        symptom = item.get('symptom')  # Extract the 'symptom' value from each item
        if symptom:
            psymptoms.append(symptom)
        else:
            if symptom==0:
                psymptoms.append(symptom)  
    
    
    models = {
    "Random Forest": rnd_forest,
    "Decision Tree": tree,
    "Gaussian Naive Bayes": nb_model,
    "Support Vector Machine": svm_model
    }

    result = run_all_models(psymptoms,models)

    return jsonify(result)











@app.route("/detect_image",methods=["POST"])
def predict_image():
    data = request.get_json()
    image_path = data.get("imgLink")

    print(image_path)
    # Load the image file
    response = requests.get(image_path)
    img = Image.open(BytesIO(response.content))
    # img = image.load_img(image_path, target_size=(224, 224))


    # // 
    img = img.resize((224, 224))

    # Convert the image to a numpy array
    img_array = image.img_to_array(img)
    
    # Normalize the image array
    img_array /= 255.0
    
    # Add a batch dimension
    img_array = np.expand_dims(img_array, axis=0)
    

    
    
    # Predict the probability across all output classes
    predictions = skin_model.predict(img_array)
    print(predictions)
    
    # Convert the probabilities to class labels
    class_index = np.argmax(predictions, axis=1)
    print(class_index)

    # Assume 'train_ds.class_names' is the list of class names in the order that the model was trained
    class_label = class_names[class_index[0]]
    
    return jsonify( get_disease_details(class_label) )

if __name__ == "__main__":
    app.run()


