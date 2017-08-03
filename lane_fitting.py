import numpy as np
from numpy.polynomial.polynomial import polyval
import cv2
from utility import *

class Line():
    '''
    Class to save traits of previous line detections
    '''
    def __init__(self):
        self.detected = False


def sliding_window(img, offset, min_pix=50, margin=100, nwindow=9, out_img=None):
    """
    get hot pixels using sliding window method on binarized image
    :param img: binarized image 
    :param offset: offset for window. 
    :param min_pix: min pixel count for altering window offset
    :param margin: left right search margin
    :param nwindow: vertical window count
    :param out_img: output img for visualization
    :return: list of x, y coordinates of hot pixels, output_image for visualization
    """
    window_height = img.shape[0] // nwindow
    # coordinates of all non zero pixels in img
    nonzero = img.nonzero()
    nonzerox = np.array(nonzero[1])
    nonzeroy = np.array(nonzero[0])

    lane_idx = list()
    if out_img is None:
        out_img = np.dstack((img, img, img)) * 255

    for window in range(nwindow):
        # get window boundaries
        bottom = img.shape[0] - (window + 1) * window_height
        top = bottom + window_height
        left = offset - margin
        right = offset + margin

        # draw rectangle for window
        cv2.rectangle(out_img, (left, bottom), (right, top), (0, 255, 0), 2)

        # find nonzero pixels inside the window
        good_idx = ((bottom <= nonzeroy) & (nonzeroy < top) &
                    (left <= nonzerox) & (nonzerox < right)).nonzero()[0]
        lane_idx.extend(good_idx)

        if len(good_idx) > min_pix:
            offset = np.int(np.mean(nonzerox[good_idx]))

    y_coords = nonzeroy[lane_idx]
    x_coords = nonzerox[lane_idx]
    return y_coords, x_coords, out_img


def car_position_offset(img_size, leftfit, rightfit, x_cvt):
    """
    get car position relative to lanes
    :param img_size:  
    :param leftfit: coefficients from numpy polyfit, left lane
    :param rightfit: coefficients from numpy polyfit, right lane
    :param x_cvt: picture to world conversion ratio
    :param y_cvt: 
    :return: i.e. ('left', 5)
    """
    bottom = img_size[0]
    lane_center = (polyval(bottom, leftfit[::-1]) + polyval(bottom, rightfit[::-1])) / 2
    img_center = img_size[1] // 2
    direction = 'left'
    if lane_center > img_center:
        direction = 'right'
    offset = abs(lane_center - img_center) * x_cvt
    return (direction, offset)


def calculate_curve(x, y, x_cvt, y_cvt, pos=720):
    fit = np.polyfit(y * y_cvt, x * x_cvt, 2)
    curve = ((1 + (2 * fit[0] * pos * y_cvt + fit[1]) ** 2) ** 1.5) \
        / np.absolute(2 * fit[0])
    return curve


def weighted_lane(origin_img, bin_img, Minv, left_fit, right_fit):
    # Create an image to draw the lines on
    warp_zero = np.zeros_like(bin_img).astype(np.uint8)
    color_warp = np.dstack((warp_zero, warp_zero, warp_zero))

    ploty = np.linspace(0, bin_img.shape[0] - 1, bin_img.shape[0])
    left_fitx = left_fit[0] * ploty ** 2 + left_fit[1] * ploty + left_fit[2]
    right_fitx = right_fit[0] * ploty ** 2 + right_fit[1] * ploty + right_fit[2]

    # Recast the x and y points into usable format for cv2.fillPoly()
    pts_left = np.array([np.transpose(np.vstack([left_fitx, ploty]))])
    pts_right = np.array([np.flipud(np.transpose(np.vstack([right_fitx, ploty])))])
    pts = np.hstack((pts_left, pts_right))

    # Draw the lane onto the warped blank image
    cv2.fillPoly(color_warp, np.int_([pts]), (0, 255, 0))

    # Warp the blank back to original image space using inverse perspective matrix (Minv)
    newwarp = cv2.warpPerspective(color_warp, Minv, (bin_img.shape[1], bin_img.shape[0]))
    # Combine the result with the original image
    result = cv2.addWeighted(origin_img, 1, newwarp, 0.3, 0)
    return result


def pipeline(img):
    """
    pipeline function for undistort, binarize, fit and output process
    :param img: BGR format
    :return: 
    """
    # Define conversions in x and y from pixels space to meters
    ym_per_pix = 30 / 720  # meters per pixel in y dimension
    xm_per_pix = 3.7 / 520  # meters per pixel in x dimension

    M, warped = get_birdview(img)
    Minv = np.linalg.inv(M)

    gray = cv2.cvtColor(warped, cv2.COLOR_BGR2GRAY)
    hls = cv2.cvtColor(warped, cv2.COLOR_BGR2HLS)

    sobel = mag_thresh(gray, thresh=(2.5,255))
    combined = np.zeros_like(sobel)
    combined[(sobel == 1) & (hls[..., 2] >= 150)] = 1

    histogram = np.sum(combined[combined.shape[0] // 2:, ...], axis=0)
    midpoint = np.int(histogram.shape[0] / 2)
    leftx_base = np.argmax(histogram[:midpoint])
    rightx_base = np.argmax(histogram[midpoint:]) + midpoint

    left_ys, left_xs, _ = sliding_window(combined, leftx_base)
    right_ys, right_xs, _ = sliding_window(combined, rightx_base)

    # temp sanity check
    if len(left_ys) == 0 or len(left_xs) == 0 or len(right_ys) == 0 or len(right_xs) == 0:
        return img

    # Fit a second order polynomial to each
    left_fit = np.polyfit(left_ys, left_xs, 2)
    right_fit = np.polyfit(right_ys, right_xs, 2)

    curvel = calculate_curve(left_xs, left_ys, xm_per_pix, ym_per_pix)
    curver = calculate_curve(right_xs, right_ys, xm_per_pix, ym_per_pix)

    off_d, off_v = car_position_offset(combined.shape, left_fit, right_fit, xm_per_pix)

    out_img = weighted_lane(img, combined, Minv, left_fit, right_fit)
    message = 'CurveL: {:.1f} CurveR: {:.1f}   {:.1f}m to {}'.format(curvel, curver, off_v, off_d)
    cv2.putText(out_img, message, (50, 50), cv2.FONT_HERSHEY_SIMPLEX, 1, (255,0,0), 2, cv2.LINE_AA)
    return out_img




