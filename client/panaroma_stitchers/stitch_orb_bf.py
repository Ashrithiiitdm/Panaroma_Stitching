import cv2 as cv
import numpy as np
import streamlit as st

def stitch_panorama(images):

    result_img = images[0]

    for i in range(1, len(images)):

        img1_gray = cv.cvtColor(result_img, cv.COLOR_RGB2GRAY)
        img2_gray = cv.cvtColor(images[i], cv.COLOR_RGB2GRAY)

        orb = cv.ORB_create(nfeatures = 2000)

        keypoints_img1, descriptors_img1 = orb.detectAndCompute(img1_gray, None)
        keypoints_img2, descriptors_img2 = orb.detectAndCompute(img2_gray, None)

        # Use BFMatcher with Hamming distance (suitable for ORB)
        bf = cv.BFMatcher(cv.NORM_HAMMING, crossCheck = False)

        matches = bf.knnMatch(descriptors_img1, descriptors_img2, k = 2)
        
        good_matches = []
        for m, n in matches:
            if m.distance < 0.6 * n.distance:
                good_matches.append(m)

        if len(good_matches) >= 4:
            src_pts = np.float32([keypoints_img1[m.queryIdx].pt for m in good_matches]).reshape(-1, 1, 2)
            dst_pts = np.float32([keypoints_img2[m.trainIdx].pt for m in good_matches]).reshape(-1, 1, 2)

            H, status = cv.findHomography(dst_pts, src_pts, cv.RANSAC, 5.0)

            if H is not None and status is not None:
                inliers = np.sum(status)
                inlier_ratio = float(inliers) / len(status)

                if inliers < 10 or inlier_ratio < 0.3:
                    st.error(f"Image {i+1}: Homography rejected due to poor inlier support ({inliers} inliers, ratio={inlier_ratio:.2f})")
                    return None
                print("H matrix+ orb_bf:", H)
                # Warp and stitch images
                h1, w1 = result_img.shape[:2]
                h2, w2 = images[i].shape[:2]

                pts = np.float32([[0, 0], [0, h2], [w2, h2], [w2, 0]]).reshape(-1, 1, 2)
                dst = cv.perspectiveTransform(pts, H)

                min_x, min_y = np.int32(dst.min(axis=0).ravel())
                max_x, max_y = np.int32(dst.max(axis=0).ravel())

                trans_x = max(0, -min_x)
                trans_y = max(0, -min_y)

                transform_array = np.array([[1, 0, trans_x], [0, 1, trans_y], [0, 0, 1]], dtype=np.float32)
                H_adjusted = transform_array @ H

                warped_img = cv.warpPerspective(images[i], H_adjusted, (max(max_x + trans_x, result_img.shape[1] + trans_x), max(max_y + trans_y, result_img.shape[0] + trans_y)))
                
                warped_result = cv.warpPerspective(result_img, transform_array, (warped_img.shape[1], warped_img.shape[0]))

                # Blend the two images
                result_img = np.where(warped_result > 0, warped_result, warped_img)
            else:
                st.error("Failed to compute homography between images.")
                return None
        else:
            st.error(f"Not enough matching points found between image {i} and {i+1}")
            return None

    return result_img
