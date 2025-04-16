import cv2 as cv
import numpy as np

def stitch_panorama(images):
    result_img = images[0]
    
    for i in range(1, len(images)):
        img1_gray = cv.cvtColor(result_img, cv.COLOR_RGB2GRAY)
        img2_gray = cv.cvtColor(images[i], cv.COLOR_RGB2GRAY)
        
        sift = cv.SIFT_create()
        keypoints_img1, descriptors_img1 = sift.detectAndCompute(img1_gray, None)
        keypoints_img2, descriptors_img2 = sift.detectAndCompute(img2_gray, None)
        
        bf = cv.BFMatcher()
        matches = bf.knnMatch(descriptors_img1, descriptors_img2, k = 2)
        
        good_matches = []
        for m, n in matches:
            if m.distance < 0.75 * n.distance:
                good_matches.append(m)
        
        if len(good_matches) >= 4:
            src_pts = np.float32([keypoints_img1[m.queryIdx].pt for m in good_matches])
            dst_pts = np.float32([keypoints_img2[m.trainIdx].pt for m in good_matches])
            
            H, status = cv.findHomography(dst_pts, src_pts, cv.RANSAC, 4.0)
            
            if H is not None:
                h1, w1 = result_img.shape[:2]
                h2, w2 = images[i].shape[:2]
                
                pts = np.float32([[0, 0], [0, h2], [w2, h2], [w2, 0]]).reshape(-1, 1, 2)
                dst = cv.perspectiveTransform(pts, H)
                
                min_x, min_y = np.int32(dst.min(axis = 0).ravel())
                max_x, max_y = np.int32(dst.max(axis = 0).ravel())
                
                trans_x = max(0, -min_x)
                trans_y = max(0, -min_y)
                transform_array = np.array([[1, 0, trans_x], [0, 1, trans_y], [0, 0, 1]], dtype = np.float32)
                
                H_adjusted = transform_array.dot(H)
                
                warped_img = cv.warpPerspective(images[i], H_adjusted, (max(max_x, w1) + trans_x, max(max_y, h1) + trans_y))
                warped_result = cv.warpPerspective(result_img, transform_array, (warped_img.shape[1], warped_img.shape[0]))
                
                result_img = np.where(warped_result > 0, warped_result, warped_img)
            else:
                raise ValueError(f"Failed to compute homography between image {i} and {i+1}")
        else:
            raise ValueError(f"Not enough matching points found between image {i} and {i+1}")
    
    return result_img
