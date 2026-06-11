"""
تست‌های لایه ML
"""

import pytest
import numpy as np
import torch
from ml_models import (
    FraudLSTM, LSTMTrainer,
    FraudAutoencoder, AutoencoderTrainer,
    BehavioralProfiler, MLScorer,
    SEQ_LEN, INPUT_DIM, BEHAVIOR_DIM, N_CLUSTERS,
)


def make_sequence(n=8):
    return np.random.rand(n, SEQ_LEN, INPUT_DIM).astype(np.float32)

def make_behavioral(n=50):
    return np.random.rand(n, BEHAVIOR_DIM).astype(np.float32)


class TestFraudLSTM:
    def test_output_shape(self):
        model = FraudLSTM()
        x = torch.rand(4, SEQ_LEN, INPUT_DIM)
        out = model(x)
        assert out.shape == (4, 1)

    def test_output_range(self):
        model = FraudLSTM()
        x = torch.rand(4, SEQ_LEN, INPUT_DIM)
        out = model(x)
        assert (out >= 0).all() and (out <= 1).all()

    def test_train_epoch_returns_loss(self):
        model   = FraudLSTM()
        trainer = LSTMTrainer(model)
        X = make_sequence(16)
        y = np.random.randint(0, 2, 16).astype(np.float32)
        loss = trainer.train_epoch(X, y, batch_size=8)
        assert isinstance(loss, float)
        assert loss >= 0

    def test_predict_proba_shape(self):
        model   = FraudLSTM()
        trainer = LSTMTrainer(model)
        X = make_sequence(5)
        probs = trainer.predict_proba(X)
        assert probs.shape == (5,)

    def test_predict_proba_range(self):
        model   = FraudLSTM()
        trainer = LSTMTrainer(model)
        X = make_sequence(5)
        probs = trainer.predict_proba(X)
        assert (probs >= 0).all() and (probs <= 1).all()


class TestAutoencoder:
    def test_output_shape(self):
        model = FraudAutoencoder()
        x = torch.rand(4, SEQ_LEN * INPUT_DIM)
        out = model(x)
        assert out.shape == (4, SEQ_LEN * INPUT_DIM)

    def test_train_epoch_returns_loss(self):
        model   = FraudAutoencoder()
        trainer = AutoencoderTrainer(model)
        X = make_sequence(16)
        loss = trainer.train_epoch(X, batch_size=8)
        assert loss >= 0

    def test_fit_threshold(self):
        model   = FraudAutoencoder()
        trainer = AutoencoderTrainer(model)
        X = make_sequence(32)
        trainer.fit_threshold(X, percentile=95)
        assert trainer.threshold is not None
        assert trainer.threshold >= 0

    def test_predict_anomaly_shape(self):
        model   = FraudAutoencoder()
        trainer = AutoencoderTrainer(model)
        X_train = make_sequence(32)
        trainer.fit_threshold(X_train)
        X_test  = make_sequence(5)
        is_anom, scores = trainer.predict_anomaly(X_test)
        assert is_anom.shape == (5,)
        assert scores.shape  == (5,)

    def test_predict_without_threshold_raises(self):
        model   = FraudAutoencoder()
        trainer = AutoencoderTrainer(model)
        X = make_sequence(4)
        with pytest.raises(ValueError):
            trainer.predict_anomaly(X)


class TestBehavioralProfiler:
    def test_fit_and_predict(self):
        profiler = BehavioralProfiler(n_clusters=N_CLUSTERS)
        X = make_behavioral(50)
        profiler.fit(X)
        cluster, name, dist = profiler.predict_profile(X[0])
        assert 0 <= cluster < N_CLUSTERS
        assert isinstance(name, str)
        assert dist >= 0

    def test_anomaly_detection(self):
        profiler = BehavioralProfiler(n_clusters=N_CLUSTERS)
        X = make_behavioral(50)
        profiler.fit(X)
        normal  = X[0]
        anomaly = np.ones(BEHAVIOR_DIM, dtype=np.float32) * 100
        _, normal_score  = profiler.is_anomalous_behavior(normal)
        _, anomaly_score = profiler.is_anomalous_behavior(anomaly)
        assert anomaly_score >= normal_score

    def test_predict_without_fit_raises(self):
        profiler = BehavioralProfiler()
        with pytest.raises(ValueError):
            profiler.predict_profile(np.zeros(BEHAVIOR_DIM))


class TestMLScorer:
    def _make_scorer(self):
        lstm_model = FraudLSTM()
        lstm       = LSTMTrainer(lstm_model)
        ae_model   = FraudAutoencoder()
        ae         = AutoencoderTrainer(ae_model)
        X_train    = make_sequence(32)
        ae.fit_threshold(X_train)
        profiler   = BehavioralProfiler(n_clusters=N_CLUSTERS)
        profiler.fit(make_behavioral(50))
        return MLScorer(lstm_trainer=lstm, ae_trainer=ae, profiler=profiler)

    def test_score_returns_dict(self):
        scorer = self._make_scorer()
        seq = make_sequence(1)
        beh = make_behavioral(1)[0]
        result = scorer.score(sequence=seq, behavioral=beh)
        assert "final_score" in result
        assert "is_anomaly"  in result

    def test_final_score_range(self):
        scorer = self._make_scorer()
        seq = make_sequence(1)
        beh = make_behavioral(1)[0]
        result = scorer.score(sequence=seq, behavioral=beh)
        assert 0.0 <= result["final_score"] <= 1.0

    def test_partial_score_no_crash(self):
        scorer = MLScorer(lstm_trainer=None, ae_trainer=None, profiler=None)
        result = scorer.score(sequence=None, behavioral=None)
        assert result["final_score"] == 0.0

    def test_high_anomaly_flagged(self):
        scorer = self._make_scorer()
        seq = make_sequence(1)
        beh = np.ones(BEHAVIOR_DIM, dtype=np.float32) * 100
        result = scorer.score(sequence=seq, behavioral=beh)
        assert result["behavioral_score"] > 0
