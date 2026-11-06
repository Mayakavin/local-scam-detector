cd ~/project
cat > report_notes.md << 'EOF'
# Report notes (ML / AOT part)

## Data
- Sources used: UCI SMS Spam Collection, SMS Phishing Dataset (Mishra & Soni, Mendeley Data, DOI 10.17632/f45bkkt8pr.1), and an Indian-context scam/legit SMS dataset (appears as "sentinel" in the results) [FILL: exact name, link, licence].
- Label definition: positive = message attempts fraud or credential theft; negative = everything else, including legitimate promotions. [FILL: how UCI spam rows and the Mendeley spam/smishing labels were handled.]
- Cleaning: lowercase labels, exact-text deduplication across sources, conflicting-label texts removed. [FILL: rows before/after, total scam ratio.]
- Splits: validation 1,530 and test 1,530 messages. [FILL: train size, vocab size, max length 160.]
- Test set composition: 930 sentinel, 514 UCI, 86 Mendeley messages. Positives: 773 of 1,530 (about 50.5%).

## Model
- Character-level CNN (embedding, 1D convolution, global max pool, linear, softmax), trained in PyTorch, FP32.
- Epoch and decision threshold selected on the validation set; threshold = 0.6.

## Results (held-out test set, threshold 0.6)
| Slice | Precision | Recall | F1 | FPR | Positives |
|---|---|---|---|---|---|
| Overall (n=1530) | 0.997 | 0.982 | 0.990 | 0.26% | 773 |
| sentinel (n=930) | 0.999 | 0.999 | 0.999 | 0.40% | 677 |
| UCI (n=514) | 0.981 | 0.823 | 0.895 | 0.22% | 62 |
| Mendeley (n=86) | 1.000 | 0.941 | 0.970 | 0.00% | 34 |
- Overall confusion matrix: TP 759, FP 2, FN 14, TN 755.
- Validation numbers (F1 0.996) are optimistic because epoch and threshold were chosen on that set. Test numbers are the ones to report.
- Earlier baseline (UCI only, v1, snapshots/v1_uci_only): test F1 0.924, precision 0.897, recall 0.953, FPR 1.5% at threshold 0.5. Note the test sets differ, so this is not a like-for-like comparison.

## ExecuTorch export
- Exported with torch.export, lowered with the XNNPACK partitioner, saved as models/sms_charcnn.pte (56,968 bytes). FP32; INT8 quantisation not applied (tiny model, FP32 already small and fast).
- Delegation: 2 delegated subgraphs, 5 delegated nodes, 5 non-delegated nodes (results/delegation_summary.txt).
- Validation of the .pte against the PyTorch model on the full test set (n=1,530): argmax agreement 1.0, threshold-decision agreement 1.0, max absolute difference 1.0e-6.
- Environment: Python 3.12, ExecuTorch 1.6.0 (see VERSIONS.md).

## Model contract for the Android app
- models/model_spec.json, models/vocab.json, models/golden_test.json. Preprocessing: lowercase, keep first 160 code points, map to vocab index (unknown = 1), pad with 0.

## Limitations
- The sentinel slice scores near-perfectly (F1 0.999). That is a warning sign that its messages may be templated or synthetic, so the overall figure is dominated by an easy slice. The UCI slice (recall 0.82) is the more realistic indicator. [FILL: result of manually reading 20 sentinel messages.]
- The test set is about 50% scams. Real phones see far fewer, so real-world precision will be lower than reported;
- Text only: no sender identity, so a genuine carrier promotion and a fake one with identical wording look the same.
- English (and Roman-script Indian English) only; Hindi/Tamil script support is future work.
EOF