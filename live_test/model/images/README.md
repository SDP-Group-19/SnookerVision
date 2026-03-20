# LabelImg Instructions

This folder contains three member subfolders: `James`, `Kasper`, and `William`.
Each member should label ONLY the images inside their own folder.

## Setup
1. Install and launch LabelImg:
   - `labelImg` or `labelimg`
   - If that fails: `python3 -m labelImg`
2. Set format to `YOLO` in the LabelImg UI.
3. Load classes file:
   - `live_test/model/classes.txt`

## For Each Member
James:
- Open images: `live_test/model/images/James`
- Save labels: `live_test/model/labels/train`

Kasper:
- Open images: `live_test/model/images/Kasper`
- Save labels: `live_test/model/labels/train`

William:
- Open images: `live_test/model/images/William`
- Save labels: `live_test/model/labels/train`

## Labeling Workflow
1. `Open Dir` and choose your images folder.
2. `Change Save Dir` and choose your labels folder.
3. Press `w` to draw a bounding box.
4. Pick the correct class.
5. Press `Ctrl+S` to save.
6. Use `a`/`d` to move between images.

## Notes
- Only label images in your own folder.
- If you don’t see the class list, re-open `classes.txt` from above.
