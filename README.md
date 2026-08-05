# YOLO11m Person Detection Pipeline

This repository contains scripts to run the YOLO11m Person Detection training pipeline, focusing on combining COCO (person class) and CrowdHuman datasets.

## Structure

*   `scripts/`: Contains the numbered Python scripts (00 to 08) that execute the pipeline end-to-end.
    *   `00_setup_environment.py`: Setup environment and download base YOLO model.
    *   `01_download_datasets.py`: Download COCO and CrowdHuman.
    *   `02_convert_coco.py`: Convert COCO annotations to YOLO format.
    *   `03_convert_crowdhuman.py`: Convert CrowdHuman annotations to YOLO format (fbox).
    *   `04_split_and_organize.py`: Split dataset (70/20/10) and generate configuration.
    *   `05_sanity_check.py`: Verify converted labels.
    *   `06_train.py`: Train YOLO11m model.
    *   `07_validate.py`: Validate model against subsets.
    *   `08_export_deliverable.py`: Export final deliverable.

*   `YOLO11m_Person_Detection_Training_Notes (3).md`: The original training notes forming the basis of this pipeline.

## Execution

Execute the scripts sequentially from `00` to `08` from the root of this project.

```bash
python scripts/00_setup_environment.py
# ... and so on
```

