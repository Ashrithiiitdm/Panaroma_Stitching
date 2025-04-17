import streamlit as st
import cv2 as cv
import numpy as np
from PIL import Image
import io
from panaroma_stitchers import stitch_sift_bf, stitch_orb_bf, stitch_sift_flann

st.set_page_config(
    page_title="Panorama Stitcher",
    page_icon="🏞️",
    layout="wide",
)

def convert_to_pil(image):
    return Image.fromarray(image)

def get_download_button(image_pil, label):
    buf = io.BytesIO()
    image_pil.save(buf, format="JPEG")
    byte_im = buf.getvalue()
    st.download_button(
        label=f"Download {label}",
        data=byte_im,
        file_name=f"{label.replace(' ', '_')}.jpg",
        mime="image/jpeg",
    )

def main():
    st.markdown(
        """
        <style>
        .title {
            text-align: center;
            font-size: 50px;
        }
        </style>
        <h1 class="title">Panorama Image Stitcher</h1>
        """, 
        unsafe_allow_html=True
    )

    st.write("""
    ## Upload your images to create a panorama
    
    Upload two or more images that have overlapping areas, and the app will stitch them 
    together into a panorama using different algorithms for comparison.
    
    ### Tips for best results:
    - Make sure images have sufficient overlap
    - Use images with similar lighting conditions
    - Upload images in the correct order (left to right)
    """)

    uploaded_files = st.file_uploader(
        "Choose images to stitch",
        type=["jpg", "jpeg", "png"],
        accept_multiple_files=True
    )

    if uploaded_files:
        if len(uploaded_files) < 2:
            st.error("Please upload at least 2 images.")
            return

        st.write("### Uploaded Images")
        cols = st.columns(min(len(uploaded_files), 4))
        for i, file in enumerate(uploaded_files):
            cols[i % 4].image(file, caption=f"Image {i+1}", use_column_width=True)

        if st.button("Stitch Using All Algorithms", type="primary"):
            with st.spinner("Processing images..."):

                # Convert uploaded files to OpenCV format
                images = []
                for file in uploaded_files:
                    image_bytes = file.getvalue()
                    nparr = np.frombuffer(image_bytes, np.uint8)
                    img = cv.imdecode(nparr, cv.IMREAD_COLOR)
                    img_rgb = cv.cvtColor(img, cv.COLOR_BGR2RGB)
                    images.append(img_rgb)

                # Stitch using different algorithms
                try:
                    stitched_results = {
                        "SIFT + BF Matcher": stitch_sift_bf.stitch_panorama(images),
                        "SIFT + FLANN Matcher": stitch_sift_flann.stitch_panorama(images),
                        "ORB + BF Matcher": stitch_orb_bf.stitch_panorama(images),
                    }

                    st.markdown("## Panorama Results (Comparison)")
                    col1, col2, col3 = st.columns(3)
                    for (name, result), col in zip(stitched_results.items(), [col1, col2, col3]):
                        if result is not None:
                            result_pil = convert_to_pil(result)
                            col.image(result_pil, caption=name, use_column_width=True)
                            with col:
                                get_download_button(result_pil, label=name)
                        else:
                            col.warning(f"{name} failed to stitch.")

                except Exception as e:
                    st.error(f"Error during stitching: {str(e)}")

    # Sidebar info
    st.sidebar.title("About")
    st.sidebar.info("""
        This app stitches multiple images into a panorama using feature-based techniques.
        It supports:
        - SIFT + BF
        - SIFT + FLANN
        - ORB + BF
    """)

    st.sidebar.title("Technical Details")
    st.sidebar.markdown("""
        - **Feature Detection**: SIFT, ORB
        - **Matching**: BF Matcher, FLANN
        - **Stitching**: Homography with RANSAC
        - **Frontend**: Streamlit
    """)

if __name__ == "__main__":
    main()
