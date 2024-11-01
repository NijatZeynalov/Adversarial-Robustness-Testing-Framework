# Adversarial Robustness Testing Framework

A comprehensive framework for testing and improving AI model robustness against adversarial attacks. Built with modular design for easy extension and integration.

## Project Structure
```
adv_robustness/
├── src/
│   ├── attacks/
│   │   ├── __init__.py
│   │   ├── fgsm.py        # Fast Gradient Sign Method
│   │   ├── pgd.py         # Projected Gradient Descent
│   │   └── cw.py          # Carlini-Wagner Attack
│   ├── defenses/
│   │   ├── __init__.py
│   │   ├── adversarial_training.py
│   │   ├── input_transformation.py
│   │   └── detection.py
│   ├── evaluation/
│   │   ├── __init__.py
│   │   ├── metrics.py
│   │   ├── visualization.py
│   │   └── reporting.py
│   ├── models/
│   │   ├── __init__.py
│   │   ├── base.py
│   │   ├── classifier.py
│   │   └── detector.py
│   ├── training/
│   │   ├── __init__.py
│   │   ├── trainer.py
│   │   └── callbacks.py
│   └── utils/
│       ├── __init__.py
│       ├── logger.py
│       └── validators.py
└── tests/
    ├── __init__.py
    ├── test_attacks.py
    ├── test_defenses.py
    └── test_evaluation.py
```

## Features

### Attacks (`src/attacks/`)
- FGSM (Fast Gradient Sign Method)
- PGD (Projected Gradient Descent)
- CW (Carlini-Wagner)

### Defenses (`src/defenses/`)
- Adversarial Training
- Input Transformations
- Attack Detection

### Evaluation (`src/evaluation/`)
- Robustness Metrics
- Result Visualization
- Report Generation

### Models (`src/models/`)
- Base Model Interface
- Classifier Implementation
- Attack Detector

## Installation

1. Clone the repository:
```bash
git clone https://github.com/yourusername/adv-robustness.git
cd adv-robustness
```

2. Create a virtual environment:
```bash
python -m venv venv
source venv/bin/activate  # On Windows: venv\Scripts\activate
```

3. Install dependencies:
```bash
pip install -r requirements.txt
```

## Usage

### Basic Attack Example
```python
from src.attacks import FGSM
from src.models import Classifier
from src.evaluation import metrics

# Initialize model and attack
model = Classifier()
attack = FGSM(epsilon=0.03)

# Generate adversarial examples
adv_examples = attack.generate(model, data, labels)

# Evaluate robustness
results = metrics.evaluate_robustness(model, adv_examples)
```

### Implementing Defenses
```python
from src.defenses import AdversarialTraining
from src.training import Trainer

# Setup defense
defense = AdversarialTraining(
    model=model,
    attack=FGSM(epsilon=0.03)
)

# Train with defense
trainer = Trainer(model=model, defense=defense)
trainer.train(train_data, epochs=10)
```

## License

MIT License - see LICENSE file for details.