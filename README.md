# BrainScanAI — local prototype

This project includes:
- `app.py`: Streamlit prototype for CNN classification, U-Net segmentation, and Qwen + LoRA chat.
- The CNN `.keras` file supplied in this conversation.
- The `BrainScanAI_NLP_v2` adapter files.
- A trained U-Net segmentation model (`models/unet/BrainScanAI_UNet_Final_v3.keras`).

## Important model compatibility notes

1. A trained U-Net is already included (`models/unet/BrainScanAI_UNet_Final_v3.keras`, 64×64 input). The app automatically loads the newest `.keras` file in `models/unet/`, so you can drop in a replacement of your own and it will be picked up on the next restart. The previously bundled, untrained model is kept in `models/unet/_archive/` for reference.
2. The app assumes the CNN has four outputs ordered as `glioma, meningioma, no tumor, pituitary`. This order was verified by running the bundled model on labeled samples, so it can be trusted for this `.keras` file. The EfficientNetB0-based model includes its own input preprocessing and therefore expects raw pixel values in `[0, 255]` — the app feeds it `[0, 255]` and must NOT divide by 255. Confirm these match the exact CNN training notebook before trusting outputs.
3. The bundled U-Net takes grayscale slices resized to 64×64 and normalized to [0,1], and emits a per-pixel sigmoid probability (threshold 0.5). The app infers the input size from the model itself, so a 128×128 or 256×256 model also works if you replace it. It was trained on the T1-CE-style `brain_tumor_kaggle` set; a random image or non-brain scan may not be compatible. Held-out test Dice: mean 0.748 / median 0.840 (481 slices).
4. The NLP adapter is not a full language model. On first chat, the app downloads `Qwen/Qwen2.5-1.5B-Instruct` from Hugging Face, so internet is needed initially. Running it on CPU may be slow and requires several GB of RAM.
5. This is a student/demo prototype, not a medical device. Model predictions are not a diagnosis; do not use for medical decisions.

## Setup on Windows

Open Command Prompt in this folder and run:

```bat
python -m venv venv
venv\Scripts\activate
python -m pip install --upgrade pip
pip install -r requirements.txt
streamlit run app.py
```

If TensorFlow/PyTorch installation fails due to Python version compatibility, use a supported 64-bit Python version (commonly Python 3.10 or 3.11) and create a fresh virtual environment.

## Folder layout

```text
BrainScanAI_Complete/
  app.py
  requirements.txt
  README.md
  .gitignore
  models/
    cnn/   (included CNN model)
    unet/  (included trained U-Net; _archive/ holds the old untrained v2)
    nlp/   (included adapter; not the Qwen base model)
  unet_previews/  (sample [input | ground truth | prediction] overlays)
  chat_history/
```

The chat history is kept in Streamlit session memory for this prototype; it is not a permanent patient record.
