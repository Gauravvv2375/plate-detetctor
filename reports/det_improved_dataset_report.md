# Improved detector dataset report

No model training was run.

## original

- Path: `/Users/gauravghanekar/Downloads/number_data/data/det`
- Format: YOLO detection
- Images / label files: 8975 / 8974
- Input/assigned splits (train/val/test): 6308 / 1784 / 883
- Original class IDs: [0]
- Declared class names: ['license_plate']
- Empty/background labels: 151
- Missing / orphan labels: 1 / 0
- Corrupt images / malformed annotations: 0 / 0
- Readable JPEGs with incomplete EOI markers: 0
- Exact duplicate images within source: 0 in 0 groups
- Final contribution (train/val/test): 6308 / 1784 / 882

## kaggle1

- Path: `/Users/gauravghanekar/Downloads/number_data/data/kaggle 1`
- Format: Pascal VOC XML (pixel xyxy)
- Images / label files: 47 / 47
- Input/assigned splits (train/val/test): 34 / 7 / 4
- Original class IDs: not applicable (Pascal VOC names)
- Declared class names: none
- Empty/background labels: 0
- Missing / orphan labels: 0 / 0
- Corrupt images / malformed annotations: 0 / 2
- Readable JPEGs with incomplete EOI markers: 7
- Exact duplicate images within source: 0 in 0 groups
- Final contribution (train/val/test): 34 / 7 / 4

## kaggle2

- Path: `/Users/gauravghanekar/Downloads/number_data/data/kaggle 2`
- Format: Pascal VOC XML (pixel xyxy; object names are plate transcriptions)
- Images / label files: 1698 / 1697
- Input/assigned splits (train/val/test): 1358 / 143 / 193
- Original class IDs: not applicable (Pascal VOC names)
- Declared class names: none
- Empty/background labels: 0
- Missing / orphan labels: 3 / 2
- Corrupt images / malformed annotations: 0 / 1
- Readable JPEGs with incomplete EOI markers: 2
- Exact duplicate images within source: 48 in 48 groups
- Final contribution (train/val/test): 1288 / 127 / 183

## kaggle3

- Path: `/Users/gauravghanekar/Downloads/number_data/data/kaggle 3`
- Format: YOLO detection
- Images / label files: 46 / 46
- Input/assigned splits (train/val/test): 10 / 8 / 28
- Original class IDs: [0]
- Declared class names: ['number_plate']
- Empty/background labels: 0
- Missing / orphan labels: 0 / 0
- Corrupt images / malformed annotations: 0 / 0
- Readable JPEGs with incomplete EOI markers: 0
- Exact duplicate images within source: 0 in 0 groups
- Final contribution (train/val/test): 10 / 8 / 28

## Final data/det_improved

- Train / val / test: 7640 / 1926 / 1097
- Total images: 10663
- Plate annotations: 10871
- Background images: 151
- Exact duplicate groups found: 48
- Non-conflicting duplicate copies skipped: 0
- Conflicting duplicate images rejected: 96 in 48 groups
- Final class: `0: license_plate`
- Readable source JPEGs with incomplete EOI markers repaired only in copied dataset: 9

## Validation

- Passed: **True**
- Label problems: 0
- YAML issues: 0
- Corrupt final images: 0
- Cross-split exact duplicate leakage: 0
- Ultralytics parsed targets: [{'split': 'train', 'images': 7640, 'targets': 7749, 'unique_class_ids': [0.0], 'finite': True, 'minimum': 0.0, 'maximum': 0.0}, {'split': 'val', 'images': 1926, 'targets': 1999, 'unique_class_ids': [0.0], 'finite': True, 'minimum': 0.0, 'maximum': 0.0}, {'split': 'test', 'images': 1097, 'targets': 1123, 'unique_class_ids': [0.0], 'finite': True, 'minimum': 0.0, 'maximum': 0.0}]

See the JSON report for exact filenames, rejected annotations, duplicate groups, and original Pascal VOC object names.

Visual samples: `reports/det_improved_visual_audit.jpg`  
Independent validator output: `reports/det_improved_validation.json`
