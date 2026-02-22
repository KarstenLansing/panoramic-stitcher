import cv2
import numpy as np
import os
import sys
import glob
import re

def natural_key(filename):
    return [int(s) if s.isdigit() else s.lower() for s in re.split(r'(\d+)', filename)]

def compute_homography_dlt(src_pts, dst_pts):
    num_pts = src_pts.shape[0]
    if num_pts < 4:
        raise ValueError("At least 4 points required")
    A = []
    for i in range(num_pts):
        x, y = src_pts[i]
        xp, yp = dst_pts[i]
        A.append([-x, -y, -1, 0, 0, 0, x * xp, y * xp, xp])
        A.append([0, 0, 0, -x, -y, -1, x * yp, y * yp, yp])
    A = np.array(A)
    U, S, Vt = np.linalg.svd(A)
    h = Vt[-1, :]
    H = h.reshape((3, 3))
    if H[2, 2] != 0:
        H = H / H[2, 2]
    return H

def compute_homography_ransac(src_pts, dst_pts, num_iters=2000, thresh=3.0):
    n_pts = src_pts.shape[0]
    if n_pts < 4:
        return None
    
    best_H = None
    best_inlier_count = 0
    best_mask = None
    
    for _ in range(num_iters):
        indices = np.random.choice(n_pts, 4, replace=False)
        src_sample = src_pts[indices]
        dst_sample = dst_pts[indices]
        try:
            H_candidate = compute_homography_dlt(src_sample, dst_sample)
        except Exception:
            continue
        src_h = np.hstack([src_pts, np.ones((n_pts, 1))])
        projected = (H_candidate @ src_h.T).T
        projected = projected / np.where(projected[:, [2]]==0, 1, projected[:, [2]])
        errors = np.linalg.norm(projected[:, :2] - dst_pts, axis=1)
        mask = errors < thresh
        inlier_count = np.sum(mask)
        if inlier_count > best_inlier_count:
            best_inlier_count = inlier_count
            best_H = H_candidate
            best_mask = mask
    if best_H is None or best_inlier_count < 4:
        return None
    
    src_inliers = src_pts[best_mask]
    dst_inliers = dst_pts[best_mask]
    
    best_H = compute_homography_dlt(src_inliers, dst_inliers)
    return best_H

def extract_features(img):
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    sift = cv2.SIFT_create()
    kp, des = sift.detectAndCompute(gray, None)
    return kp, des

def match_features(des1, des2, ratio_thresh=0.7):
    bf = cv2.BFMatcher()
    matches = bf.knnMatch(np.asarray(des1, dtype=np.float32), np.asarray(des2, dtype=np.float32), k=2)
    good = [m for m, n in matches if m.distance < ratio_thresh * n.distance]
    return good

def get_matched_points(kp1, kp2, matches):
    pts1 = np.float32([kp1[m.queryIdx].pt for m in matches])
    pts2 = np.float32([kp2[m.trainIdx].pt for m in matches])
    return pts1, pts2

def my_warp_perspective(img, H, dsize):
    h, w = dsize[1], dsize[0]
    xx, yy = np.meshgrid(np.arange(w), np.arange(h))
    
    ones = np.ones_like(xx, dtype=np.float32)
    dest_coords = np.stack([xx, yy, ones], axis=-1).reshape(-1, 3).T
    invH = np.linalg.inv(H)
    
    src_coords = invH @ dest_coords
    src_coords /= np.where(src_coords[2, :] == 0, 1, src_coords[2, :])
    
    map_x = src_coords[0, :].reshape(h, w).astype(np.float32)
    map_y = src_coords[1, :].reshape(h, w).astype(np.float32)
    
    return cv2.remap(img, map_x, map_y, interpolation=cv2.INTER_LINEAR, borderMode=cv2.BORDER_CONSTANT)

def warp_and_blend(base_img, next_img, H):
    h1, w1 = base_img.shape[:2]
    h2, w2 = next_img.shape[:2]
    corners_next = np.float32([[0,0], [0,h2], [w2,h2], [w2,0]]).reshape(-1,1,2)
    warped_corners = []
    
    for pt in corners_next:
        p = H @ np.array([pt[0][0], pt[0][1], 1])
        p = p / (p[2] if p[2] != 0 else 1)
        warped_corners.append(p[:2])
        
    warped_corners = np.array(warped_corners)
    corners_base = np.float32([[0,0], [0,h1], [w1,h1], [w1,0]]).reshape(-1,1,2)
    all_corners = np.concatenate((warped_corners.reshape(-1,2), corners_base.reshape(-1,2)), axis=0)
    
    xmin = int(np.floor(np.min(all_corners[:,0])))
    ymin = int(np.floor(np.min(all_corners[:,1])))
    xmax = int(np.ceil(np.max(all_corners[:,0])))
    ymax = int(np.ceil(np.max(all_corners[:,1])))
    
    translation = [-xmin, -ymin]
    H_trans = np.array([[1, 0, translation[0]],
                        [0, 1, translation[1]],
                        [0, 0, 1]], dtype=np.float32)
    dsize = (xmax - xmin, ymax - ymin)
    warped_img = my_warp_perspective(next_img, H_trans.dot(H), dsize)
    
    canvas = np.zeros((dsize[1], dsize[0], 3), dtype=np.float32)
    mask_base = np.zeros((dsize[1], dsize[0]), dtype=np.float32)
    mask_warped = np.zeros((dsize[1], dsize[0]), dtype=np.float32)
    
    canvas[translation[1]:h1+translation[1], translation[0]:w1+translation[0]] = base_img.astype(np.float32)
    mask_base[translation[1]:h1+translation[1], translation[0]:w1+translation[0]] = 1.0
    
    warped_gray = cv2.cvtColor(warped_img, cv2.COLOR_BGR2GRAY)
    mask_warped[warped_gray > 0] = 1.0
    warped_img = warped_img.astype(np.float32)
    
    mask_base_3c = cv2.merge([mask_base, mask_base, mask_base])
    mask_warped_3c = cv2.merge([mask_warped, mask_warped, mask_warped])
    
    combined = canvas * mask_base_3c + warped_img * mask_warped_3c
    weight = mask_base_3c + mask_warped_3c
    weight[weight == 0] = 1.0
    
    blended = combined / weight
    blended = np.clip(blended, 0, 255).astype(np.uint8)
    
    return blended

def stitch_outwards(images):
    n = len(images)
    center_idx = n // 2
    panorama = images[center_idx]
    
    
    left_limit = max(0, center_idx - 5)
    for i in range(center_idx - 1, left_limit - 1, -1):
        kp_pan, des_pan = extract_features(panorama)
        kp_img, des_img = extract_features(images[i])
        matches = match_features(des_img, des_pan, ratio_thresh=0.7)
        if len(matches) < 4:
            continue
        pts_img, pts_pan = get_matched_points(kp_img, kp_pan, matches)
        H = compute_homography_ransac(pts_img, pts_pan, num_iters=2000, thresh=3.0)
        if H is None:
            continue
        panorama = warp_and_blend(panorama, images[i], H)
        
        
    right_limit = min(n, center_idx + 6)
    for i in range(center_idx + 1, right_limit):
        kp_pan, des_pan = extract_features(panorama)
        kp_img, des_img = extract_features(images[i])
        matches = match_features(des_img, des_pan, ratio_thresh=0.7)
        if len(matches) < 4:
            continue
        pts_img, pts_pan = get_matched_points(kp_img, kp_pan, matches)
        H = compute_homography_ransac(pts_img, pts_pan, num_iters=2000, thresh=3.0)
        if H is None:
            continue
        panorama = warp_and_blend(panorama, images[i], H)
        
        
    return panorama

def stitch_directory(directory, output_path="panorama.jpg"):
    image_paths = []
    for ext in ("*.jpg", "*.JPG"):
        image_paths.extend(glob.glob(os.path.join(directory, ext)))
    if not image_paths:
        return None
    image_paths.sort(key=natural_key)
    total = len(image_paths)
    center = total // 2
    start = max(0, center - 5)
    end = min(total, center + 6)
    selected_paths = image_paths[start:end]
    images = []
    for path in selected_paths:
        img = cv2.imread(path)
        if img is not None:
            images.append(img)
    if len(images) < 2:
        return None
    panorama = stitch_outwards(images)
    cv2.imwrite(output_path, panorama)
    return panorama

if __name__ == "__main__":
    if len(sys.argv) > 1:
        input_dir = sys.argv[1]

    final_panorama = stitch_directory(input_dir, output_path="final_panorama.jpg")