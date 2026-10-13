# classroom-violence-fall-detection
classroom-violence-fall-detection/
│
├── README.md
├── requirements.txt
├── .gitignore
├── LICENSE
│
├── configs/
│   ├── model.yaml
│   └── dataset.yaml
│
├── data/
│   ├── README.md
│   ├── raw/
│   ├── processed/
│   ├── annotations/
│   └── metadata/
│
├── notebooks/
│   ├── data_exploration.ipynb
│   ├── pose_extraction.ipynb
│   └── model_evaluation.ipynb
│
├── src/
│   ├── data/
│   │   ├── extract_frames.py
│   │   ├── create_clips.py
│   │   └── split_dataset.py
│   │
│   ├── pose/
│   │   ├── pose_detector.py
│   │   ├── tracker.py
│   │   └── feature_extractor.py
│   │
│   ├── models/
│   │   ├── lstm_model.py
│   │   ├── rule_based.py
│   │   └── train.py
│   │
│   ├── evaluation/
│   │   ├── metrics.py
│   │   └── evaluate.py
│   │
│   └── inference/
│       ├── predict_video.py
│       └── alert_manager.py
│
├── app/
│   └── app.py
│
├── tests/
├── models/
├── outputs/
│   ├── videos/
│   ├── logs/
│   └── figures/
│
└── docs/
    ├── system_design.md
    ├── labeling_guide.md
    ├── dataset_card.md
    └── responsible_ai.md
