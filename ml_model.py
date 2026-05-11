import os
import json
import torch
import joblib
from sklearn.ensemble import RandomForestClassifier


class MultiUserModel:

    def __init__(self, model_path="model/multi_user_model.pkl"):
        self.model = RandomForestClassifier(n_estimators=150)
        self.trained = False
        self.model_path = model_path

        # Try loading existing model
        self.load_model()

    # ----------------------------
    # Convert dict → feature vector
    # ----------------------------
    def dict_to_features(self, data):
        feat_to_idx = {
            "hold_1": 0, "press_press": 1, "release_press": 2,
            "release_release": 3, "hold_2": 4, "total_time": 5,
            "slope_h1": 6, "slope_pp": 7, "slope_rp": 8,
            "slope_rr": 9, "slope_h2": 10, "slope_tt": 11
        }

        try:
            tensor = torch.zeros(len(data), 12)

            for digram in data:
                for feature in data[digram]:
                    if feature in feat_to_idx:
                        tensor[int(digram)][feat_to_idx[feature]] = float(data[digram][feature])

            return torch.mean(tensor, dim=0).tolist()

        except Exception as e:
            print(" Feature extraction error:", e)
            return [0] * 12

    # ----------------------------
    # Load all users dataset
    # ----------------------------
    def load_all_users(self):
        X = []
        y = []

        base_path = "dataset/passphrase"

        if not os.path.exists(base_path):
            print(" Dataset folder not found")
            return X, y

        for file in os.listdir(base_path):
            if file.endswith(".json"):

                try:
                    username = file.split("_")[0]
                    file_path = os.path.join(base_path, file)

                    with open(file_path) as f:
                        samples = json.load(f)

                    for sample in samples:
                        features = self.dict_to_features(sample)

                        if sum(features) != 0:  # avoid empty data
                            X.append(features)
                            y.append(username)

                except Exception as e:
                    print(f" Error loading {file}:", e)

        return X, y

    # ----------------------------
    # Train model
    # ----------------------------
    def train(self):
        X, y = self.load_all_users()

        if len(X) == 0:
            print(" No training data found")
            return False

        self.model.fit(X, y)
        self.trained = True

        # Save model
        self.save_model()

        print(f" Model trained on {len(X)} samples and saved")
        return True

    # ----------------------------
    # Predict user
    # ----------------------------
    def predict_user(self, probe):

        # Ensure trained
        if not self.trained:
            print(" Model not trained, training now...")
            if not self.train():
                return "No Model", 0

        # Handle list input
        if isinstance(probe, list):
            if len(probe) == 0:
                return "No Data", 0
            probe = probe[0]

        # Validate input
        if not isinstance(probe, dict):
            print(" Invalid probe:", type(probe), probe)
            return "Error", 0

        try:
            features = self.dict_to_features(probe)

            prediction = self.model.predict([features])[0]
            probabilities = self.model.predict_proba([features])[0]

            confidence = max(probabilities) * 100

            return prediction, round(confidence, 2)

        except Exception as e:
            print(" Prediction error:", e)
            return "Error", 0

    # ----------------------------
    # Save model
    # ----------------------------
    def save_model(self):
        try:
            os.makedirs(os.path.dirname(self.model_path), exist_ok=True)
            joblib.dump(self.model, self.model_path)
            print(" Model saved")

        except Exception as e:
            print(" Model save failed:", e)

    # ----------------------------
    # Load model
    # ----------------------------
    def load_model(self):
        if os.path.exists(self.model_path):
            try:
                self.model = joblib.load(self.model_path)
                self.trained = True
                print(" Model loaded from disk")

            except Exception as e:
                print(" Failed to load model:", e)