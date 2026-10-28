# FashionForge AI — Generative Fashion Studio

A Streamlit interface for the uploaded FashionForge AI Conditional GAN generator.

## Project structure

```text
FashionForgeAI_App/
├── app.py
├── requirements.txt
├── README.md
└── models/
    ├── FashionForgeAI_Generator.h5
    └── category_mapping.json
```

## Important: verify the category mapping

The uploaded model has **8 category labels**, but the `.h5` file does not contain the original human-readable `class_names` list from your training notebook. The included `category_mapping.json` uses a provisional category list based on the dataset filtering plan.

Before relying on category-specific output, replace `class_names` and `class_to_id` in `models/category_mapping.json` with the exact values printed by your Colab training notebook:

```python
print(class_names)
print(class_to_id)
```

The order must exactly match the IDs used during training. A wrong mapping will not stop the app from running, but the selected label may represent a different category than its displayed name.

## Run locally on Windows

Install a compatible Python version for your TensorFlow release. Then open Command Prompt in this folder and run:

```bat
py -m venv .venv
.venv\Scripts\activate
python -m pip install --upgrade pip
pip install -r requirements.txt
streamlit run app.py
```

The app opens at `http://localhost:8501`.

## What it includes

- Category selector
- Adjustable number of generated concepts
- Repeatable random seed
- Image gallery
- Individual PNG downloads
- Download-all ZIP
- Cached model loading

## Notes

- The generator creates 128 × 128 RGB images.
- Generated images are experimental concepts and may contain artifacts.
- The app loads the provided HDF5 model for inference only; it does not resume training.
- Only load model files you trust.
