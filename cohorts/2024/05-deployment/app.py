from flask import Flask, request, jsonify
import pickle


app = Flask(__name__)

with open("homework/dv.bin", "rb") as f:
    dv = pickle.load(f)

with open("homework/model1.bin", "rb") as f:
    model = pickle.load(f)


@app.route("/predict", methods=["POST"])
def predict():
    client = request.json
    X_client = dv.transform([client])
    probability = model.predict_proba(X_client)[0, 1]
    return jsonify({"probability": probability})


if __name__ == "__main__":
    app.run(host="0.0.0.0", port=5000)
