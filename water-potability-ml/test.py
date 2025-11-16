import requests
from sklearn.feature_extraction import DictVectorizer

url = "http://localhost:9696/predict"

dv = DictVectorizer(sparse=False)
sample = {
    "ph": 5.584086638456089,
    "Hardness": 188.3133237696164,
    "Solids": 28748.68773904612,
    "Chloramines": 7.54486878877965,
    "Sulfate": 393.66339551509645,
    "Conductivity": 280.4679159334877,
    "Organic_carbon": 8.399734640152758,
    "Trihalomethanes": 54.917861841994466,
    "Turbidity": 2.5597082275565217,
}
X = dv.fit_transform([sample])
response = requests.post(url, json=sample)
predictions = response.json()

print(predictions)
