# AML Fraud Detection System

## Overview

AML Fraud Detection System is an intelligent Anti-Money Laundering (AML) platform designed to identify suspicious financial activities and potentially fraudulent transactions using a combination of rule-based analysis, graph analytics, and machine learning models.

The system processes transaction data, extracts behavioral features, applies risk scoring mechanisms, and generates alerts for suspicious entities and activities.

---

## Key Features

* Rule-based risk detection engine
* Machine Learning–based suspicious activity detection
* Graph analysis for relationship and network investigation
* Transaction feature engineering pipeline
* Risk scoring and prioritization
* Automated alert generation
* Comprehensive test suite
* Docker deployment support
### Dashboard Features

The dashboard provides an interactive interface for AML investigations and risk monitoring.

Key capabilities include:

* Viewing account risk status and risk scores
* Monitoring high-risk and blocked accounts
* Investigating suspicious entities through a centralized dashboard
* Accessing account holder information
* Reviewing risk factors and reasons behind account blocking
* Viewing timestamps and historical events related to suspicious activities
* Supporting analysts in compliance and fraud investigation workflows

---

## Architecture

### Data Ingestion

Collects and validates transaction records from supported data sources.

### Feature Engineering

Generates behavioral and transactional features used by the detection models.

### Risk Engine

Calculates risk scores based on predefined AML rules and model outputs.

### Machine Learning Models

Applies trained ML models to identify unusual transaction patterns and suspicious behavior.

### Graph Analysis

Builds transaction networks and identifies hidden relationships between entities.

### Dashboard

Provides analytical insights and visualization for investigators.

---

## Project Structure

```text
.
├── dashboard_v2.py
├── features.py
├── graph.py
├── ingestion.py
├── ml_models.py
├── risk_engine.py
├── rules.py
├── docker-compose.yml
├── models/
├── tests/
└── README.md
```

---

## Installation

### Clone Repository

```bash
git clone https://github.com/avid8/aml-fraud-detection.git
cd aml-fraud-detection
```

### Create Virtual Environment

```bash
python3 -m venv venv
source venv/bin/activate
```

### Install Dependencies

```bash
pip install -r requirements.txt
```

---

## Running the Project

```bash
python dashboard_v2.py
```

Or run individual modules depending on the workflow:

```bash
python ingestion.py
python risk_engine.py
python ml_models.py
```

---

## Testing

Run all tests:

```bash
pytest
```

Run specific tests:

```bash
pytest test_risk_engine.py
pytest test_ml_models.py
```

---

## Technologies

* Python
* Machine Learning
* Graph Analytics
* Docker
* PyTest

---

## Disclaimer

This project is intended for educational, research, and demonstration purposes. It does not constitute a complete production AML compliance solution.

---

## Author

Mehdi

