# E-Rara Image Matcher

A comprehensive tool for searching and matching historical book illustrations from e-rara using IIIF image APIs and advanced computer vision techniques.

## Description

This project provides tools for matching illustrations in historical books using computer vision and deep learning techniques. It downloads images from e-rara's IIIF API and uses various matching algorithms to find similar illustrations.

## Features

- **GUI Interface**: Easy-to-use graphical interface for non-technical users
- **Search Configuration**: Query e-rara's database by author, title, publisher, place, and date range
- **Advanced Image Matching**: Use state-of-the-art deep learning models (SuperPoint, LoFTR, XFeat, SIFT)
- **Parallel Processing**: Efficient batch matching with configurable worker threads
- **Progress Tracking**: Real-time progress bars and status updates
- **GPU Support**: Optional CUDA acceleration for faster matching
- **Configuration Management**: Save and load search/matching configurations
- **Results Export**: Export matching results and logs

## Installation

Install with uv (recommended):

```bash
# Clone the repository
git clone https://github.com/BoiMat/image_matcher.git
cd image_matcher

# Basic installation
uv sync

# With GPU support (requires CUDA)
uv sync --extra gpu

# With development tools
uv sync --extra dev
```

## Quick Start

### Using the GUI (Recommended)

Launch the graphical interface:

```bash
uv run gui_app.py
```

The GUI provides three main tabs for a complete workflow:

## GUI Usage Guide

### Tab 1: Search Configuration

Configure your search parameters to find relevant historical documents:

1. **Search Parameters**:
   - **Author**: Search by author name (e.g., "Goethe")
   - **Title**: Search by book title (e.g., "Historia")
   - **Publisher**: Search by publisher name
   - **Place**: Search by publication place (e.g., "Bern", "Basel")
   - **Date Range**: Set from/until dates (e.g., 1600-1700)
   - **Max Records**: Limit number of results (leave empty for all)

2. **Search Options**:
   - **Website Style Search**: Use e-rara's web interface search style (recommended)
   - **Truncation**: Enable/disable search term truncation

3. **Actions**:
   - **Search IDs**: Execute the search and generate IDs file
   - **Load IDs from File**: Load previously saved IDs
   - **Save/Load Config**: Save current settings for reuse

4. **Status**: Real-time updates showing search progress and results

### Tab 2: Image Matching

Configure image matching parameters and run the matching process:

1. **Target Image**:
   - **Browse**: Select your reference image to match against
   - Supports: JPG, PNG, BMP, TIFF formats

2. **Output Settings**:
   - **Output Directory**: Choose where to save matched images

3. **Matching Parameters**:
   - **Matcher**: Choose algorithm:
     - `superpoint-lg`: Best quality, requires GPU (recommended)
     - `sift-nn`: Good quality, works on CPU
   - **Device**: Select `cpu` or `cuda` (auto-detects availability)
   - **Min Matches**: Minimum inliers required (auto-adjusts based on matcher)
   - **Max Workers**: Parallel processing threads (5 recommended)

4. **Processing Options**:
   - **Apply Preprocessing**: Remove borders from images
   - **Expand to Pages**: Process individual pages instead of just covers

5. **Input IDs**:
   - **IDs File**: Select file with record IDs (from Tab 1 or manual file)

### Tab 3: Results & Logs

View and manage results from your matching operations:

1. **Results Display**: Complete log of all operations and results
2. **Actions**:
   - **Open Output Folder**: Open folder with downloaded matches
   - **Save Log**: Export log to text file
   - **Clear**: Clear the results display

## Workflow Example

1. **Set up Search** (Tab 1):
   - Enter "Bern" in Place field
   - Set dates: 1600-1620
   - Set max records: 50
   - Click "Search IDs"

2. **Configure Matching** (Tab 2):
   - Browse and select your target image
   - Choose matcher: `superpoint-lg` (if GPU available) or `sift-nn`
   - Verify IDs file is loaded from Tab 1
   - Click "Run Image Matching"

3. **View Results** (Tab 3):
   - Monitor progress in real-time
   - Review matching results and download counts
   - Open output folder to see matched images

## Configuration Management

### Save Configuration
Save your current search and matching settings:
- Go to Tab 1 → "Save Search Config"
- Choose filename (e.g., `my_search.json`)
- Settings include all search parameters, matching options, and file paths

### Load Configuration
Restore previously saved settings:
- Go to Tab 1 → "Load Search Config" 
- Select your saved `.json` file
- All tabs will be populated with saved settings

### Example Configuration
```json
{
  "search_filters": {
    "place": "Bern",
    "from_date": "1600",
    "until_date": "1620",
    "max_records": 50
  },
  "matching_config": {
    "matcher": "superpoint-lg",
    "device": "cuda",
    "min_matches": "100"
  },
  "paths": {
    "target_image": "path/to/your/image.jpg",
    "output_dir": "results"
  }
}
```

## Command Line Usage

For advanced users and automation:

### Search for Record IDs
```bash
python e_rara_id_fetcher.py --place "Bern" --from-date 1600 --until-date 1620
```

### Run Image Matching
```bash
python e_rara_image_downloader.py ids.txt target_image.jpg --matcher superpoint-lg --device cuda
```

## Matcher Recommendations

- **SuperPoint-LG**: Best accuracy, requires GPU, 100+ inliers threshold
- **SIFT-NN**: Good accuracy, works on CPU, 20+ inliers threshold
- **LoFTR**: Good for low-texture images, requires GPU
- **XFeat-LG**: Fast processing, moderate accuracy

## Tips for Best Results

1. **Image Quality**: Use high-quality, well-lit target images
2. **Preprocessing**: Enable border removal for scanned documents
3. **Thresholds**: Start with default min_matches, adjust if needed
4. **GPU Usage**: Use GPU matchers for better performance and accuracy
5. **Parallel Processing**: Increase max_workers for faster processing (but watch memory usage)

## Troubleshooting

- **No GPU Available**: Switch to `sift-nn` matcher with `cpu` device
- **Out of Memory**: Reduce max_workers or switch to CPU
- **No Matches Found**: Lower min_matches threshold or try different matcher
- **Slow Processing**: Increase max_workers or use GPU acceleration

# Citations and Acknowledgments

## Image Matching Models

This project uses the image-matching-models repository for deep learning-based image matching algorithms.

**Repository:** https://github.com/alexstoken/image-matching-models

Please cite the EarthMatch paper if you use this tool in your research:

```bibtex
@InProceedings{Berton_2024_EarthMatch,
    author    = {Berton, Gabriele and Goletto, Gabriele and Trivigno, Gabriele and Stoken, Alex and Caputo, Barbara and Masone, Carlo},
    title     = {EarthMatch: Iterative Coregistration for Fine-grained Localization of Astronaut Photography},
    booktitle = {Proceedings of the IEEE/CVF Conference on Computer Vision and Pattern Recognition (CVPR) Workshops},
    month     = {June},
    year      = {2024},
}
```

## E-Rara Digital Library

This tool accesses historical documents from the e-rara digital library:

**Website:** https://www.e-rara.ch/
**About:** e-rara is the digital platform for rare books from Swiss institutions

Please acknowledge e-rara when using their content in publications.