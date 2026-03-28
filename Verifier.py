import json
import os
import torch
import math
import random
from sklearn.ensemble import RandomForestClassifier


class Verifier:

    def __init__(self, username, input_word):
        self.username = username
        self.input_word = input_word

        self.feat_to_idx = {
            "hold_1": 0, "press_press": 1, "release_press": 2,
            "release_release": 3, "hold_2": 4, "total_time": 5,
            "slope_h1": 6, "slope_pp": 7, "slope_rp": 8,
            "slope_rr": 9, "slope_h2": 10, "slope_tt": 11
        }

    # ----------------------------
    # Convert dict → tensor
    # ----------------------------
    def dict_to_tensor(self, data):
        tot_tensor = torch.zeros(len(data), 12)

        for digram in data:
            for feature in data[digram]:
                if feature in self.feat_to_idx:
                    feat = self.feat_to_idx[feature]
                    tot_tensor[int(digram)][feat] = data[digram][feature]

        return tot_tensor

    # ----------------------------
    # Extract ML features
    # ----------------------------
    def extract_features(self, tensor):
        return torch.mean(tensor, dim=0).tolist()

    # ----------------------------
    # Naive Bayes scoring
    # ----------------------------
    def compute_logits(self, data, sample):
        means = torch.mean(data, 0)
        std = torch.std(data, 0)

        std = torch.where(std < 1e-3, torch.tensor(1e-3), std)

        score_tensor = self.gaussian_likelihood(sample, means, std)
        return torch.sum(score_tensor).item()

    def gaussian_likelihood(self, sample, mean, std):

        raw_like = 1 / ((2 * math.pi * std) ** 0.5) * \
            torch.exp(-(sample - mean) ** 2 / (2 * std))

        like = torch.nan_to_num(raw_like, nan=0)

        dummy = torch.full_like(like, 0.01)
        adjusted = torch.maximum(like, dummy)

        return torch.log(adjusted)

    # ----------------------------
    # MAIN FUNCTION (ML + Hybrid)
    # ----------------------------
    def compare_metrics(self, probe):

        word = self.input_word
        file_path = f'dataset/password/{self.username}_{word}.json'

        print("Looking for file:", file_path)

        if not os.path.exists(file_path):
            print("Dataset NOT found:", file_path)
            return False, 0, 0, 0

        # Load dataset
        with open(file_path) as f:
            data_metrics = json.load(f)

        print(f"Loaded {len(data_metrics)} samples")

        probe_tensor = self.dict_to_tensor(probe)

        l2_scores = []
        nb_scores = []

        #  Compare with all samples
        for sample in data_metrics:

            sample_tensor = self.dict_to_tensor(sample)

            min_len = min(sample_tensor.shape[0], probe_tensor.shape[0])

            sample_tensor = sample_tensor[:min_len, :]
            probe_trim = probe_tensor[:min_len, :]

            differences = torch.abs(sample_tensor - probe_trim)
            l2 = torch.norm(differences).item() / max(1, (len(self.input_word) - 1))

            nb = self.compute_logits(
                sample_tensor.unsqueeze(0),
                probe_trim
            ) / max(1, (len(self.input_word) - 1))

            l2_scores.append(l2)
            nb_scores.append(nb)

        print("All L2:", l2_scores)
        print("All NB:", nb_scores)

        best_l2 = min(l2_scores)
        best_nb = max(nb_scores)

        print("Best L2:", best_l2)
        print("Best NB:", best_nb)

        l2_threshold = sum(l2_scores) / len(l2_scores)
        nb_threshold = sum(nb_scores) / len(nb_scores)

        print("L2 Threshold:", l2_threshold)
        print("NB Threshold:", nb_threshold)

        # ============================
        #  MACHINE LEARNING PART
        # ============================

        model = RandomForestClassifier(n_estimators=50)

        features = []
        labels = []

        # Genuine samples
        for sample in data_metrics:
            t = self.dict_to_tensor(sample)
            features.append(self.extract_features(t))
            labels.append(1)

        # Fake samples (noise)
        for sample in data_metrics:
            t = self.dict_to_tensor(sample)
            noisy = t + torch.randn_like(t) * 150
            features.append(self.extract_features(noisy))
            labels.append(0)

        # Train model
        model.fit(features, labels)

        # Predict probe
        probe_features = self.extract_features(probe_tensor)
        ml_prediction = model.predict([probe_features])[0]

        print("ML Prediction:", ml_prediction)

        # ============================
        #  FINAL DECISION
        # ============================

        MAX_L2 = 3.5

        decision = (
            (ml_prediction == 1) and
            (best_l2 < l2_threshold) and
            (best_l2 < MAX_L2) and
            (best_nb >= nb_threshold)
        )

        # Confidence
        if l2_threshold > 0:
            confidence = max(0, 1 - (best_l2 / l2_threshold))
        else:
            confidence = 0

        print("Confidence:", confidence)

        return decision, best_nb, best_l2, round(confidence * 100, 2)