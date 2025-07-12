from flask import Flask, request, jsonify
from model_utils import predict_demand

app = Flask(__name__)

@app.route('/predict', methods=['POST'])
def predict():
    data = request.json
    station_id = data.get("station_id")
    timestamp = data.get("timestamp")

    if not station_id or not timestamp:
        return jsonify({"error": "Missing station_id or timestamp"}), 400

    result, error = predict_demand(station_id, timestamp)
    if error:
        return jsonify({"error": error}), 400

    category = result["category"]
    bookings = result["predicted_bookings"]
    return f"{category},{bookings}"



if __name__ == "__main__":
    app.run(debug=True)

@app.route('/', methods=['GET'])
def home():
    return "🎉 Demand Predictor is live. Use POST /predict to get started."

