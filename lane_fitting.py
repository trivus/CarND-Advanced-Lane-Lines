import numpy as np
import cv2
from utility import *
from collections import deque


def sliding_window(img, offsets, min_pix=2, margin=70, nwindow=9, out_img=None):
    """
    get hot pixels using sliding window method on binarized image
    :param img: binarized image 
    :param min_pix: min pixel count for altering window offset
    :param margin: left right search margin
    :param nwindow: vertical window count
    :param out_img: output img for visualization
    :param offsets: previous offset of each windows
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

    new_offsets = []

    for window in range(nwindow):
        # get window boundaries
        bottom = img.shape[0] - (window + 1) * window_height
        top = bottom + window_height
        left = offsets[window] - margin
        right = offsets[window] + margin

        # draw rectangle for window
        cv2.rectangle(out_img, (left, bottom), (right, top), (0, 255, 0), 2)

        # find nonzero pixels inside the window
        good_idx = ((bottom <= nonzeroy) & (nonzeroy < top) &
                    (left <= nonzerox) & (nonzerox < right)).nonzero()[0]
        lane_idx.extend(good_idx)

        if len(good_idx) > min_pix and np.std(nonzerox[good_idx]) < 50:
            offset = np.int(np.mean(nonzerox[good_idx]))
            new_offsets.append(offset)
        else:
            new_offsets.append(offsets[window])

        if len(offsets) < window + 2:
            offsets.append(new_offsets[-1])

    y_coords = nonzeroy[lane_idx]
    x_coords = nonzerox[lane_idx]
    return y_coords, x_coords, out_img, new_offsets


def car_position_offset(img_size, leftfit, rightfit, x_cvt):
    """
    get car position relative to lanes
    :param img_size:  
    :param leftfit: coefficients from numpy polyfit, left lane
    :param rightfit: coefficients from numpy polyfit, right lane
    :param x_cvt: picture to world conversion ratio
    :param y_cvt: 
    :return: offset value. Sign + is left
    """
    if leftfit is None or rightfit is None:
        return 0
    bottom = img_size[0]
    lane_center = (np.polyval(leftfit, bottom) + np.polyval(rightfit, bottom)) / 2
    img_center = img_size[1] // 2
    offset = (lane_center - img_center) * x_cvt
    return offset


def calculate_curve(x, y, x_cvt, y_cvt, pos=720):
    if len(x) == 0:
        return 0
    fit = np.polyfit(y * y_cvt, x * x_cvt, 2)
    curve = ((1 + (2 * fit[0] * pos * y_cvt + fit[1]) ** 2) ** 1.5) \
        / np.absolute(2 * fit[0])
    return curve


def weighted_lane(origin_img, bin_img, Minv, left_fit, right_fit):
    if left_fit is None or right_fit is None:
        return origin_img
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


def debug_img(result_img, gray, s_channel, binarized):
    """`
    Combine img output during process for debugging purposes
    :param result_img: original output image
    :param gray: gray scaled image
    :param s_channel: s channel from HLS format
    :param binarized: the result of binarization 
    :return: debug image
    """
    output = np.zeros((1080, 1280, 3), dtype=np.uint8)
    output[0:720, 0:1280, :] = cv2.resize(result_img, (1280, 720))
    output[-240:, :320, :] = cv2.resize(cv2.cvtColor(gray, cv2.COLOR_GRAY2RGB), (320, 240))
    output[-240:, 320:640, :] = cv2.resize(cv2.cvtColor(s_channel, cv2.COLOR_GRAY2RGB), (320, 240))
    output[-240:, 640:960, :] = cv2.resize(cv2.cvtColor(binarized, cv2.COLOR_GRAY2RGB), (320, 240))
    return output


def remove_shadow(img):
    yuv = cv2.cvtColor(img, cv2.COLOR_BGR2YUV)
    clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8))
    yuv[..., 0] = clahe.apply(yuv[..., 0])
    yuv = cv2.cvtColor(yuv, cv2.COLOR_YUV2BGR)
    return yuv


class Line:
    '''
    Class to save traits of each line detection.
    Includes simple sanity check.
    '''
    def __init__(self, x_offset=None, fit=None, n_buffer=5):
        # recent bottom x offset
        self.x_offset = x_offset
        # polynomial fit
        self.fit = fit
        self.allxs = deque(maxlen=n_buffer)
        self.allys = deque(maxlen=n_buffer)
        self.bad_frame_count = 0

    def update(self, xs, ys, offset):
        if len(xs) < 5:
            self.bad_frame_count += 1
            if self.bad_frame_count > 10:
                self.allxs.clear()
                self.allys.clear()

        else:
            self.bad_frame_count = 0
            self.allxs.append(xs)
            self.allys.append(ys)
            self.x_offset = offset

    def get_points(self):
        getx, gety = [], []
        for x in self.allxs:
            getx.extend(x)
        for y in self.allys:
            gety.extend(y)
        return getx, gety


    # TODO: enhance sanity.
    def sanity(self):
        return len(self.allxs) > 0

    def get_fit(self):
        if self.sanity() is True:
            xs, ys = self.get_points()
            self.fit = np.polyfit(ys, xs, 2)
        return self.fit


class continuous_pipeline:
    """
    wrapper class of pipeline for movie processing 
    """
    def __init__(self, binarize_function, n_buffer=5):
        self.rline = Line(n_buffer=n_buffer)
        self.lline = Line(n_buffer=n_buffer)
        self.binarize_function = binarize_function

    def pipeline(self, img, debug=True):
        """
        pipeline function for undistort, binarize, fit and output process
        :param img: RGB format
        :return: 
        """
        # Define conversions in x and y from pixels space to meters
        ym_per_pix = 30 / 720  # meters per pixel in y dimension
        xm_per_pix = 3.7 / 500  # meters per pixel in x dimension

        # remove shadow
        #img = remove_shadow(img)

        # transform to bird view
        M, warped = get_birdview(img)
        Minv = np.linalg.inv(M)

        # binarize
        gray = cv2.cvtColor(warped, cv2.COLOR_RGB2GRAY)
        hls = cv2.cvtColor(warped, cv2.COLOR_RGB2HLS)

        combined = self.binarize_function(gray, hls)

        # initialize window position
        if self.lline.x_offset is not None:
            leftx_base = self.lline.x_offset
            rightx_base = self.rline.x_offset
        else:
            # find lane
            histogram = np.sum(combined[combined.shape[0] // 2:, ...], axis=0)
            midpoint = np.int(histogram.shape[0] / 2)
            leftx_base = [np.argmax(histogram[:midpoint])]
            rightx_base = [np.argmax(histogram[midpoint:]) + midpoint]

        left_ys, left_xs, _, l_offsets = sliding_window(combined, leftx_base)
        right_ys, right_xs, _, r_offsets = sliding_window(combined, rightx_base)


        curvel = calculate_curve(left_xs, left_ys, xm_per_pix, ym_per_pix)
        curver = calculate_curve(right_xs, right_ys, xm_per_pix, ym_per_pix)

        # sanity checks
        #if min(np.subtract(r_offsets, l_offsets)) < 200 or max(np.subtract(r_offsets, l_offsets)) > 800:
        #    left_xs = []
        #    left_ys = []
        #    right_xs = []
        #    right_ys = []


        self.lline.update(left_xs, left_ys, l_offsets)
        self.rline.update(right_xs, right_ys, r_offsets)

        left_fit = self.lline.get_fit()
        right_fit = self.rline.get_fit()

        # weighted mean curve based on pixel count
        curve = (curvel * len(left_xs) + curver * len(right_xs)) / (len(left_xs) + len(right_xs) + 1)
        offset = car_position_offset(combined.shape, left_fit, right_fit, xm_per_pix)
        out_img = weighted_lane(img, combined, Minv, left_fit, right_fit)

        # get offset direction
        off_d = 'right' if offset < 0 else 'left'
        message = 'Curve: {:.1f}  {:.1f}m to {}'.format(curve, abs(offset), off_d)
        cv2.putText(out_img, message, (50, 50), cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 0, 0), 2, cv2.LINE_AA)

        # DEBUG
        if debug:
            out_img = debug_img(out_img, gray, hls[..., 2], combined * 255)
        return out_img
