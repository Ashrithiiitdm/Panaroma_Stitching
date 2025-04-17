# Panorama Image Stitcher 🏞️

This project is a **Panorama Image Stitcher** application built using **Streamlit**. It allows users to upload multiple images with overlapping areas and stitches them into a seamless panorama using different feature-based algorithms for comparison.

Check out the live demo at [Panorama Image Stitcher](https://panoramafusion.streamlit.app/).

## Features

- **Algorithms Supported**:
  - SIFT + BF Matcher
  - SIFT + FLANN Matcher
  - ORB + BF Matcher
- **Interactive UI**:
  - Upload multiple images
  - View stitched results side-by-side
  - Download stitched panoramas
- **Technical Highlights**:
  - Feature detection using SIFT and ORB
  - Matching using BF Matcher and FLANN
  - Homography estimation with RANSAC
  - Image blending for seamless results

## Installation

1. Clone the repository:

   ```bash
   git clone https://github.com/Ashrithiiitdm/Panaroma_Stitching.git
   cd Panaroma_Stitching
   ```

2. Install dependencies:

   ```bash
   pip install -r requirements.txt
   ```

3. Run the application:
   ```bash
   streamlit run client/main.py
   ```

## Usage

1. Open the application in your browser (default: `http://localhost:8501`).
2. Upload two or more images with overlapping areas.
3. Click the "Stitch Using All Algorithms" button to generate panoramas using different algorithms.
4. View the results and download the stitched panoramas.

## Tips for Best Results

- Ensure images have sufficient overlap.
- Use images with similar lighting conditions.
- Upload images in the correct order (left to right).

## Project Structure

```
DIP/
├── client/
│   ├── __init__.py
│   ├── main.py                # Streamlit application
│   └── panaroma_stitchers/    # Stitching algorithms
│       ├── stitch_orb_bf.py   # ORB + BF Matcher
│       ├── stitch_sift_bf.py  # SIFT + BF Matcher
│       ├── stitch_sift_flann.py # SIFT + FLANN Matcher
├── requirements.txt           # Python dependencies
├── README.md                  # Project documentation
```

## Dependencies

- **Python**: 3.10+
- **Libraries**:
  - `fastapi`
  - `uvicorn`
  - `opencv-python-headless`
  - `numpy`
  - `streamlit`
  - `pillow`
  - `requests`

## Acknowledgments

- OpenCV for image processing
- Streamlit for the interactive UI
- Contributors and the open-source community
