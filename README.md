# Mosaic Image Generator

[Click Me! Interactive Image Mosaic Generator ](https://huggingface.co/spaces/pranudeep555/Image-Mosaic-Generator)


A Python-based tool that generates mosaic art by recreating a target image using a grid of smaller tile images (from datasets such as CIFAR-100, custom folders, or user-selected collections).

## Features

- Converts any input image into a detailed photo mosaic  
- Uses color averaging to match tiles to target pixels  
- Supports any tile dataset (e.g., CIFAR-100, custom images)  
- Adjustable tile size and mosaic resolution  
- Saves the final mosaic as a high-quality image file  

## Tech Stack

- Python 3.8+
- NumPy — matrix operations and color computations  
- Pillow (PIL) — image processing  
- Matplotlib — visualization and preview  
- (Optional) scikit-image or OpenCV for advanced blending  

## Structure

## Project Structure
```
mosaic-image-generator/
├── src/
│   ├── mosaic_generator.py      — Core logic for mosaic creation
│   ├── utils.py                 — Helper functions (color matching, etc.)
│   └── dataset_loader.py        — Loads and preprocesses tile images
│
├── data/
│   ├── tiles/                   — Folder containing tile images
│   └── target/                  — Folder for target images
│
├── output/
│   └── final_mosaic.png         — Saved mosaic results
│
├── requirements.txt             — Dependencies
└── README.md                    — Documentation
```
## Installation

Clone the repository and install dependencies:
```bash
git clone https://github.com/pranudeepmetuku10/mosaic-image-generator.git
cd mosaic-image-generator
pip install -r requirements.txt
