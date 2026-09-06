# Module 2: model training

`train_model.py` is the supported config-driven CVAE training entry point. The
GSE141115 release configuration uses 50 epochs, batch size 512, AdamW learning
rate 5e-5 and weight decay 1e-4. Training writes new files below `work/` and
does not overwrite the locked checkpoints in `model_weights/`.

`prophead_GSE141115_architecture_reference.py` preserves the GSE141115
architecture and loss implementation from the 20260610 workflow after replacing
workstation paths with release-relative paths. It is not the recommended entry
point and does not make historical bitwise retraining possible.
