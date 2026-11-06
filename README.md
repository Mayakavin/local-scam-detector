cat > README.md << 'EOF'
# On-device scam SMS detector (ML / AOT part)

A character-level CNN that classifies SMS text as scam or normal, trained in PyTorch and exported to ExecuTorch (`.pte`, XNNPACK backend) so it can run on an Arm-based Android phone with no cloud dependency.

## What's in this repo
| Folder / file | Contents |
|---|---|
| `train/` | Data preparation, training, error analysis, spot check |
| `export/` | Model spec, ExecuTorch export and validation, golden test, Llama test cases |
| `models/` | Files for the Android app: `sms_charcnn.pte`, `model_spec.json`, `vocab.json`, `golden_test.json` (and `sms_charcnn.pt`, the PyTorch weights) |
| `results/` | Metrics, export validation, delegation summary |
| `snapshot.sh` | Saves a copy of the current model, results and scripts |
| `VERSIONS.md` | Exact package versions (Python 3.12, ExecuTorch 1.6.0) |
| `report_notes.md` | Numbers and decisions for the technical report |

## Using the model in the Android app
Everything the app needs is in `models/`:
1. Load `sms_charcnn.pte` with the ExecuTorch Android API.
2. Preprocess text as described in `model_spec.json`: lowercase, keep the first 160 **code points**, map each to its index in `vocab.json` (unknown = 1), pad with 0 to length 160. Input is an `int64` tensor of shape `[1, 160]`.
3. Output is `[normal, scam]` probabilities (softmax is inside the model). Flag as scam if `output[1] >= threshold` (threshold is in `model_spec.json`).
4. Verify your preprocessing with `golden_test.json`: it contains inputs, expected token IDs and expected outputs. Check token IDs first, then outputs.

## Results
Held-out test set, threshold 0.6 (full numbers in `results/metrics.json`):

| Slice | n | Precision | Recall | F1 | FPR |
|---|---|---|---|---|---|
| Real-data slices (UCI + Mendeley) | 600 | 0.988 | 0.865 | 0.922 | 0.2% |
| Overall | 1,530 | 0.997 | 0.982 | 0.990 | 0.26% |

The overall figure is inflated by a templated, near-synthetic Indian scam dataset that makes up most of the test set, so the real-data row is the one to rely on. See Limitations.

ExecuTorch export: `models/sms_charcnn.pte` is 56,968 bytes (FP32). It matches the PyTorch model on all 1,530 test messages (agreement 1.0, max absolute difference 1e-6). XNNPACK delegation: 5 nodes delegated, 5 not delegated (`results/delegation_summary.txt`).

## Reproduce
```bash
python3 -m venv venv && source venv/bin/activate
pip install executorch pandas scikit-learn numpy
# put the datasets in data/ (see below), then:
python train/prep_data.py        # [FILL: or the name of your pooled prep script]
python train/train.py
python export/make_spec.py
python export/export_pte.py
python export/make_golden.py
```

## Datasets
Not redistributed here. Download them and place them in `data/`:
- UCI SMS Spam Collection (Almeida et al.)
- SMS Phishing Dataset, Mishra & Soni, Mendeley Data, DOI 10.17632/f45bkkt8pr.1
- [FILL: name, link and licence of the Indian-context scam dataset]

The Mendeley dataset is largely a copy of UCI, so duplicates are removed across sources.

## Limitations
- Label definition: positive = attempts fraud or credential theft; legitimate promotions count as normal. A text-only model cannot tell a real carrier offer from a fake one with the same wording.
- English and Roman-script Indian English only.
- The test set is about 50% scams, far more than real traffic, so real-world false-alarm rates will differ from the figures above.
EOF