import os
import numpy as np
import cv2
import matplotlib.pyplot as plt
import math
from PIL import Image
from matplotlib.patches import Polygon


AU = {0: 'Inner Brow Raiser',
      1: 'Outer Brow Raiser',
      2: 'Brow Lowerer',
      3: 'Upper Lid Raiser',
      4: 'Cheek Raiser',
      5: 'Lid Tightener',
      6: 'Nose Wrinkler',
      7: 'Upper Lip Raiser',
      8: 'Lip Corner Puller',
      9: 'Dimpler',
      10: 'Lip Corner Depressor',
      11: 'Chin Raiser',
      12: 'Lip Stretcher',
      13: 'Lip Tightener',
      14: 'Lip pressor',
      15: 'Lips Part',
      16: 'Jaw Drop',
      17: 'Eyes Closed',
      18: 'Mouth Stretcher',
      19: 'Lower Lip Depressor',
      20: 'mouth',
      21: 'eye',
      22: 'whole face'
      }

EXP2AU = {
    'Anger': [2, 3, 5, 7, 11, 13, 14, 15, 16],
    'Disgust': [6, 3, 19, 11, 15, 16],
    'Fear': [0, 1, 2, 3, 12, 15, 16, 18],
    # 'Happiness': [4, 8, 15],
    'Happiness': [4, 5, 7, 8, 15],
    'Sadness': [0, 2, 4, 10, 11],
    'Surprise':[0, 1, 3, 16, 18]
}

CL_KEY = {
    'Neutral':0,
    'Happiness':1,
    'Sadness':2,
    'Surprise':3,
    'Fear':4,
    'Disgust':5,
    'Anger':6,
    # 'Contempt':7
}

def fast_draw_landmarks(img: np.ndarray,
                        heatmap: np.ndarray,
                        ldmarks: list,
                        wfp
                        ):

    x_h = [x[0] for x in ldmarks]
    y_w = [x[1] for x in ldmarks]

    alpha = 1.
    markersize = 4.5
    lw = 1.
    color = 'r'
    markeredgecolor = 'red'

    try:
        height, width = img.size
    except:
        height, width = img.shape[:2]
    # just for plotting
    # avoid plotting right over thr border. dots spill outside the image.
    x_h = [max(min(x, height - 1), 0) for x in x_h]
    y_w = [max(min(y, width - 1), 0) for y in y_w]

    fig, axes = plt.subplots(nrows=1, ncols=2, squeeze=False)
    # axes[0, 0].imshow(img[:, :, ::-1])
    axes[0, 0].imshow(img, cmap='gray')

    nums = [0, 17, 22, 27, 31, 36, 42, 48, 60, 68]

    # close eyes and mouths
    plot_close = lambda i1, i2: axes[0, 0].plot([x_h[i1], x_h[i2]],
                                                [y_w[i1], y_w[i2]],
                                                color=color,
                                                lw=lw,
                                                alpha=alpha - 0.1
                                                )
    plot_close(41, 36)
    plot_close(47, 42)
    plot_close(59, 48)
    plot_close(67, 60)

    for ind in range(len(nums) - 1):
        l, r = nums[ind], nums[ind + 1]
        axes[0, 0].plot(x_h[l:r], y_w[l:r], color=color, lw=lw,
                        alpha=alpha - 0.1)

        axes[0, 0].plot(x_h[l:r], y_w[l:r], marker='o', linestyle='None',
                        markersize=markersize, color=color,
                        markeredgecolor=markeredgecolor, alpha=alpha)

    axes[0, 1].imshow(heatmap)
    plt.subplots_adjust(top=1, bottom=0, right=1, left=0, hspace=0, wspace=0)
    for ax in fig.axes:
        ax.axis('off')
        ax.margins(0, 0)
        ax.xaxis.set_major_locator(plt.NullLocator())
        ax.yaxis.set_major_locator(plt.NullLocator())

    # plt.show()

    plt.savefig(wfp, pad_inches=0, bbox_inches='tight', dpi=300)
    plt.close()
def plot_action_units_ellipsoid(au: int,
                                h: int,
                                w: int,
                                lndmks: list,
                                ):
    assert isinstance(lndmks, list), type(lndmks)
    assert len(lndmks) == 68, len(lndmks)
    # lndmks: [(x, y), ....]: x: width, y: height.


    att_map = np.zeros((h, w))
    cp = att_map.copy()
    col = (255, 255, 255)
    a = 0  # angle
    s = 0  # start angle
    e = 360  # end angle
    f = cv2.FILLED

    if au == 22:  # whole face
        lower_face = [lndmks[i] for i in range(16, 0, -1)]
        upper_face = [lndmks[i] for i in range(17, 27)]
        upper_face = [[x, y - 20] for x , y in upper_face]
        face_lds = [ld for ld in lower_face] + [ld for ld in upper_face]
        face_lds = np.array(face_lds)
        points = face_lds.reshape((-1, 1, 2))
        cv2.fillPoly(att_map, [points], color=col)

    elif au == 21: #eye
        l_x1, _ = lndmks[36]
        _, l_y1 = lndmks[38]
        r_x1, _ = lndmks[39]
        _, r_y1 = lndmks[41]
        l_x2, _ = lndmks[42]
        _, l_y2 = lndmks[44]
        r_x2, _ = lndmks[45]
        _, r_y2 = lndmks[47]

        x = int((l_x1 + r_x1) / 2)
        y = int((l_y1 + r_y1) / 2)
        major = max(int((r_x1 - l_x1) / 2), 0) + 7
        minor = max(int((r_y1 - l_y1) / 2), 0) + 7

        if major == 0:
            major = 5

        if minor == 0:
            minor = 5

        cv2.ellipse(att_map, (x, y), (major, minor), a, s, e, col, f)

        x = int((l_x2 + r_x2) / 2)
        y = int((l_y2 + r_y2) / 2)
        major = max(int((r_x2 - l_x2) / 2), 0) + 7
        minor = max(int((r_y2 - l_y2) / 2), 0) + 7

        if major == 0:
            major = 5

        if minor == 0:
            minor = 5

        cv2.ellipse(att_map, (x, y), (major, minor), a, s, e, col, f)

    elif au == 20:  # Mouth
        l_x, _ = lndmks[48]
        r_x, _ = lndmks[54]
        _, l_y = lndmks[51]
        _, r_y = lndmks[57]

        x = int((l_x + r_x) / 2)
        y = int((l_y + r_y) / 2)
        major = max(int((r_x - l_x) / 2), 0) + 7
        minor = max(int((r_y - l_y) / 2), 0) + 7

        if major == 0:
            major = 5

        if minor == 0:
            minor = 5

        cv2.ellipse(att_map, (x, y), (major, minor), a, s, e, col, f)

    elif au == 0:  # AU 0: inner brow raiser
        l_x1, l_y1 = lndmks[20]
        r_x2, r_y2 = lndmks[23]
        major = round(w / 8)
        minor = round(h/10)
        cv2.ellipse(att_map, (l_x1, l_y1), (major, minor), a, s, e, col, f)
        cv2.ellipse(att_map, (r_x2, r_y2), (major, minor), a, s, e, col, f)

    elif au == 1:  # AU 1: Outer Brow Raiser

        l_x1, l_y1 = lndmks[18]
        r_x2, r_y2 = lndmks[25]
        major = round(w / 8)
        minor = round(h / 10)

        cv2.ellipse(att_map, (l_x1, l_y1), (major, minor), a, s, e, col, f)
        cv2.ellipse(att_map, (r_x2, r_y2), (major, minor), a, s, e, col, f)

    elif au == 2:  # AU 2: Brow Lowerer
        l_x, l_y = lndmks[19]
        r_x, r_y = lndmks[24]
        x = int((l_x + r_x) / 2)
        y = int((l_y + r_y) / 2)
        # major = abs(int((r_x - l_x) / 2.))
        # minor = abs(int((r_y - l_y) / 2.))
        major = max(int((r_x - l_x) / 2), 0)
        minor = max(int((r_y - l_y) / 2), 0)

        if major == 0:
            major = 10

        if minor == 0:
            minor = 10

        cv2.ellipse(att_map, (x, y), (major, minor), a, s, e, col, f)

    elif au == 3:  # AU 3: Upper Lid Raiser
        l_x1, _ = lndmks[36]
        _, l_y1 = lndmks[38]
        r_x1, _ = lndmks[39]
        _, r_y1 = lndmks[41]
        l_x2, _ = lndmks[42]
        _, l_y2 = lndmks[44]
        r_x2, _ = lndmks[45]
        _, r_y2 = lndmks[47]

        x = int((l_x1 + r_x1) / 2)
        y = int((l_y1 + r_y1) / 2)
        major = max(int((r_x1 - l_x1) / 2), 0)
        minor = max(int((r_y1 - l_y1) / 2), 0)

        if major == 0:
            major = 5

        if minor == 0:
            minor = 5

        cv2.ellipse(att_map, (x, y), (major, minor), a, s, e, col, f)

        x = int((l_x2 + r_x2) / 2)
        y = int((l_y2 + r_y2) / 2)
        major = max(int((r_x2 - l_x2) / 2), 0)
        minor = max(int((r_y2 - l_y2) / 2), 0)

        if major == 0:
            major = 5

        if minor == 0:
            minor = 5

        cv2.ellipse(att_map, (x, y), (major, minor), a, s, e, col, f)

    elif au == 4:  # AU 4: Cheek Raiser
        major = round(w / 15)
        minor = round(h / 10)

        l_x1, l_y1 = lndmks[48]
        r_x1, r_y1 = lndmks[31]
        x = round((l_x1 + r_x1) / 2)
        y = round((l_y1 + r_y1) / 2)
        angle = math.atan2(r_y1 - l_y1, r_x1 - l_x1)
        angle = -round(math.degrees(angle)) - 30

        cv2.ellipse(att_map, (x - 5, y), (major, minor), angle, s, e, col, f)

        l_x1, l_y1 = lndmks[35]
        r_x1, r_y1 = lndmks[54]
        x = round((l_x1 + r_x1) / 2)
        y = round((l_y1 + r_y1) / 2)

        angle = math.atan2(r_y1 - l_y1, r_x1 - l_x1)
        angle = - round(math.degrees(angle)) + 20

        cv2.ellipse(att_map, (x + 5, y), (major, minor), angle, s, e, col, f)

    elif au == 5: # AU 5: Lid Tightener
        l_x1, _ = lndmks[36]
        _, l_y1 = lndmks[38]
        r_x1, _ = lndmks[39]
        _, r_y1 = lndmks[41]
        l_x2, _ = lndmks[42]
        _, l_y2 = lndmks[44]
        r_x2, _ = lndmks[45]
        _, r_y2 = lndmks[47]

        x = int((l_x1 + r_x1) / 2)
        y = int((l_y1 + r_y1) / 2)
        major = max(int((r_x1 - l_x1) / 2), 0)
        minor = max(int((r_y1 + 10 - l_y1 + 10) / 2), 0)

        if major == 0:
            major = 5

        if minor == 0:
            minor = 5

        cv2.ellipse(att_map, (x, y), (major, minor), a, s, e, col, f)

        x = int((l_x2 + r_x2) / 2)
        y = int((l_y2 + r_y2) / 2)
        major = max(int((r_x2 - l_x2) / 2), 0)
        minor = max(int((r_y2 + 10 - l_y2 + 10) / 2), 0)

        if major == 0:
            major = 5

        if minor == 0:
            minor = 5

        cv2.ellipse(att_map, (x, y), (major, minor), a, s, e, col, f)

    elif au == 6:  # AU 6: Nose Wrinkler
        l_x1, l_y1 = lndmks[29]
        r_x1, r_y1 = lndmks[31]
        r_x2, r_y2 = lndmks[35]

        cv2.ellipse(att_map, (r_x1, l_y1), (20, 20), a, s, e, col, f)
        cv2.ellipse(att_map, (r_x2, l_y1), (20, 20), a, s, e, col, f)

    elif au == 7:  # AU 7: Upper Lip Raiser
        l_x1, l_y1 = lndmks[48]
        l_x2, l_y2 = lndmks[50]

        r_x1, r_y1 = lndmks[52]
        r_x2, r_y2 = lndmks[54]

        x = int((l_x1 + l_x1) / 2)
        y = int((l_y2 + l_y2) / 2)

        cv2.ellipse(att_map, (x, y), (20, 20), a, s, e, col, f)

        x = int((r_x1 + r_x1) / 2)
        y = int((r_y2 + r_y2) / 2)

        cv2.ellipse(att_map, (x, y), (20, 20), a, s, e, col, f)

    elif au == 8:  # AU 8: Lip corner puller.
        l_x, l_y = lndmks[48]
        r_x, r_y = lndmks[54]

        major = 20
        minor = 20

        cv2.ellipse(att_map, (l_x, l_y), (major, minor), a, s, e, col, f)
        cv2.ellipse(att_map, (r_x, r_y), (major, minor), a, s, e, col, f)

    elif au == 9:  # AU 9: Dimpler.
        l_x, l_y = lndmks[48]
        r_x, r_y = lndmks[54]

        major = 20
        minor = 20

        l_x = max(l_x - 20 , 0)
        r_x = max(min((r_x + 20, w)), min((r_x + 10, w)))

        cv2.ellipse(att_map, (l_x, l_y), (major, minor), a, s, e, col, f)
        cv2.ellipse(att_map, (r_x, r_y), (major, minor), a, s, e, col, f)

    elif au == 10:  # AU 10 == AU 8: Lip Corner Depressor
        l_x, l_y = lndmks[48]
        r_x, r_y = lndmks[54]

        major = 20
        minor = 20

        cv2.ellipse(att_map, (l_x, l_y), (major, minor), a, s, e, col, f)
        cv2.ellipse(att_map, (r_x, r_y), (major, minor), a, s, e, col, f)

    elif au == 11:  # AU 11: Chin Raiser
        l_x, l_y = lndmks[59]
        r_x, r_y = lndmks[8]

        x = int((l_x + r_x) / 2)
        y = int((l_y + r_y) / 2)

        major = max(int((r_x - l_x) / 2), 0)
        minor = max(int((r_y - l_y) / 2), 0)

        if major == 0:
            major = 5

        if minor == 0:
            minor = 5

        cv2.ellipse(att_map, (x, y), (major, minor), a, s, e, col, f)


    elif au == 12:  # AU 12: Lip Stretcher
        l_x1, l_y1 = lndmks[48]
        l_x2, l_y2 = lndmks[6]


        r_x1, r_y1 = lndmks[50]
        r_x2, r_y2 = lndmks[10]

        x = int((l_x1 + l_x2) / 2)
        y = int((l_y1 + l_y2) / 2)
        major = 20
        minor = 20

        cv2.ellipse(att_map, (x, y), (major, minor), a, s, e, col, f)

        x = int((r_x1 + r_x2) / 2)
        y = int((r_y1 + r_y2) / 2)

        cv2.ellipse(att_map, (x, y), (major, minor), a, s, e, col, f)


    elif au == 13:  # AU 13: Lip Tightener
        l_x, _ = lndmks[48]
        r_x, _ = lndmks[54]
        _, l_y = lndmks[51]
        _, r_y = lndmks[57]

        x = int((l_x + r_x) / 2)
        y = int((l_y + r_y) / 2)

        major = max(int((r_x - l_x) / 2), 0)
        minor = max(int((r_y - l_y) / 2), 0)

        if major == 0:
            major = 5

        if minor == 0:
            minor = 5

        cv2.ellipse(att_map, (x, y), (major, minor), a, s, e, col, f)


    elif au == 14:  # AU 14: Lip pressor == AU 13.
        l_x, _ = lndmks[48]
        r_x, _ = lndmks[54]
        _, l_y = lndmks[51]
        _, r_y = lndmks[57]

        x = int((l_x + r_x) / 2)
        y = int((l_y + r_y) / 2)

        major = max(int((r_x - l_x) / 2), 0)
        minor = max(int((r_y - l_y) / 2), 0)

        if major == 0:
            major = 5

        if minor == 0:
            minor = 5

        cv2.ellipse(att_map, (x, y), (major, minor), a, s, e, col, f)

    elif au == 15:  # AU 15: Lips Part

        t_x, t_y = lndmks[51]
        b_x, b_y = lndmks[57]

        major = 25
        minor = 10

        cv2.ellipse(att_map, (t_x, t_y), (major, minor), a, s, e, col, f)
        cv2.ellipse(att_map, (b_x, b_y), (major, minor), a, s, e, col, f)

    elif au == 16:  # AU 16: Jaw Drop
        l_x, _ = lndmks[48]
        r_x, _ = lndmks[54]
        _, l_y = lndmks[51]
        _, r_y = lndmks[57]

        x = int((l_x + r_x) / 2)
        y = int((l_y + r_y) / 2)
        major = max(int((r_x - l_x) / 2), 0)
        minor = max(int((r_y - l_y) / 2), 0)

        if major == 0:
            major = 5

        if minor == 0:
            minor = 5

        cv2.ellipse(att_map, (x, y), (major, minor), a, s, e, col, f)

    elif au == 17:  # AU 17: Eyes Closed
        l_x1, _ = lndmks[36]
        _, l_y1 = lndmks[38]
        r_x1, _ = lndmks[39]
        _, r_y1 = lndmks[41]
        l_x2, _ = lndmks[42]
        _, l_y2 = lndmks[44]
        r_x2, _ = lndmks[45]
        _, r_y2 = lndmks[47]

        x = int((l_x1 + r_x1) / 2)
        y = int((l_y1 + r_y1) / 2)

        major = max(int((r_x1 - l_x1) / 2), 0)
        minor = max(int((r_y1 + 10 - l_y1 + 10) / 2), 0)

        cv2.ellipse(att_map, (x, y), (major, minor), a, s, e, col, f)

        x = int((l_x2 + r_x2) / 2)
        y = int((l_y2 + r_y2) / 2)

        major = max(int((r_x2 - l_x2) / 2), 0)
        minor = max(int((r_y2 + 10 - l_y2 + 10) / 2), 0)
        cv2.ellipse(att_map, (x, y), (major, minor), a, s, e, col, f)

    elif au == 18:  # AU 18: Mouth stretcher == AU 16.
        l_x, _ = lndmks[48]
        r_x, _ = lndmks[54]
        _, l_y = lndmks[51]
        _, r_y = lndmks[57]

        x = int((l_x + r_x) / 2)
        y = int((l_y + r_y) / 2)
        major = max(int((r_x - l_x) / 2), 0)
        minor = max(int((r_y - l_y) / 2), 0)

        if major == 0:
            major = 5

        if minor == 0:
            minor = 5

        cv2.ellipse(att_map, (x, y), (major, minor), a, s, e, col, f)

    elif au == 19:  # AU 19: Lower lip depressor
        l_x, l_y = lndmks[59]
        r_x, r_y = lndmks[55]

        major = 20
        minor = 20

        cv2.ellipse(att_map, (l_x, l_y), (major, minor), a, s, e, col, f)
        cv2.ellipse(att_map, (r_x, r_y), (major, minor), a, s, e, col, f)

    is_roi = ((att_map - cp).sum() > 0)  # sometimes ellipse may draw outside
    # image leading to empty heatmap. this heatmap should be flagged.

    # if is_roi:
    #     att_map = cv2.resize(att_map, dsize=(64, 64))
    #
    # else:  # flag invalid heatmaps.
    #     att_map = np.zeros((64, 64)) + np.inf

    return att_map, is_roi

def fast_draw_heatmap(img: np.ndarray,
                      heatmap: np.ndarray,
                      cl: str,
                      wfp: str,
                      binary_roi: np.ndarray = None,
                      img_msk_black: np.ndarray = None,
                      img_msk_avg: np.ndarray = None,
                      img_msk_blur: np.ndarray = None,
                      diff: np.ndarray = None
                      ):

    alpha = 1.
    markersize = 4.5
    lw = 1.
    color = 'r'
    markeredgecolor = 'red'
    try:
        height, width = img.size
    except:
        height, width = img.shape[:2]

    ncols = 3

    if binary_roi is not None:
        ncols += 1

    if img_msk_black is not None:
        ncols += 1

    if img_msk_avg is not None:
        ncols += 1

    if img_msk_blur is not None:
        ncols += 1

    fig, axes = plt.subplots(nrows=1, ncols=ncols, squeeze=False)

    fontsize = 7 if ncols == 2 else 3

    i = 0
    # axes[0, i].imshow(img[:, :, ::-1])
    axes[0, i].imshow(img)
    i += 1

    # axes[0, i].imshow(img[:, :, ::-1])
    axes[0, i].imshow(img)
    axes[0, i].imshow(heatmap, alpha=0.3)
    axes[0, i].text(
        3, 40, cl,
        fontsize=fontsize,
        bbox={'facecolor': 'white', 'pad': 1, 'alpha': 0.8, 'edgecolor':'none'}
    )

    i += 1


    axes[0, i].imshow(heatmap, alpha=0.7)
    if diff is not None:
        axes[0, i].imshow(diff, alpha=0.3)
    axes[0, i].text(
        3, 40, "diff heatmap",
        fontsize=fontsize,
        bbox={'facecolor': 'white', 'pad': 1, 'alpha': 0.8, 'edgecolor': 'none'}
    )

    i += 1

    if binary_roi is not None:
        axes[0, i].imshow(binary_roi.astype(np.uint8) * 255, cmap='gray')
        axes[0, i].text(
            3, 40, 'ROI',
            fontsize=fontsize,
            bbox={'facecolor': 'white', 'pad': 1, 'alpha': 0.8,
                  'edgecolor':'none'}
        )
        i += 1

    if img_msk_black is not None:
        axes[0, i].imshow(img_msk_black[:, :, ::-1])
        axes[0, i].text(
            3, 40, 'Black masking',
            fontsize=fontsize,
            bbox={'facecolor': 'white', 'pad': 1, 'alpha': 0.8,
                  'edgecolor':'none'}
        )
        i += 1

    if img_msk_avg is not None:
        axes[0, i].imshow(img_msk_avg[:, :, ::-1])
        axes[0, i].text(
            3, 40, 'Average masking',
            fontsize=fontsize,
            bbox={'facecolor': 'white', 'pad': 1, 'alpha': 0.8,
                  'edgecolor':'none'}
        )
        i += 1

    if img_msk_blur is not None:
        axes[0, i].imshow(img_msk_blur[:, :, ::-1])
        axes[0, i].text(
            3, 40, 'Gaussian blur masking',
            fontsize=fontsize,
            bbox={'facecolor': 'white', 'pad': 1, 'alpha': 0.8,
                  'edgecolor':'none'}
        )
        i += 1



    plt.subplots_adjust(top=1, bottom=0, right=1, left=0, hspace=0, wspace=0)
    for ax in fig.axes:
        ax.axis('off')
        ax.margins(0, 0)
        ax.xaxis.set_major_locator(plt.NullLocator())
        ax.yaxis.set_major_locator(plt.NullLocator())

    # plt.show()

    plt.savefig(wfp, pad_inches=0, bbox_inches='tight', dpi=300)
    plt.close()

def facial_mask(landmark: list,
                img_size: tuple):
    ld = [(int(x), int(y)) for x, y in landmark]
    w, h = img_size
    au_ls = []
    whole_face, _ = plot_action_units_ellipsoid(au=22, h=h, w=w, lndmks=ld)
    for au in [4, 20, 21]:
        att_map, is_roi = plot_action_units_ellipsoid(au=au, h=h, w=w, lndmks=ld)
        au_ls.append(att_map)
    union_au_heatmap = np.maximum.reduce(au_ls)
    with_face_au_heatmap = np.minimum.reduce([union_au_heatmap, whole_face])
    mask = np.expand_dims(with_face_au_heatmap, axis=2)
    mask = mask / 255
    return mask

dataset = 'CelebA'
prj = '240418_acgan'
ep = str(90)
direction = 'XY'
input_path = f'/media/ziyi/glory/logs_pin/{dataset}/{prj}/results/ep{ep}/{direction}'
output_path = os.path.join('/home/glory/projects/NTU_Parkinson_Project/results', dataset, prj+'_ep'+ep, direction, 'au_mask')
landmark_path = os.path.join('/home/glory/projects/NTU_Parkinson_Project/results', dataset, prj+'_ep'+ep, direction, 'landmark')
use_au = [3, 4, 5, 10, 15]

if __name__=='__main__':
    os.makedirs(output_path, exist_ok=True)

    image_paths = []
    landmark_paths = []
    diff_paths = []
    for dirpath, dirnames, filenames in os.walk(input_path):
        for file in filenames:
            if file.endswith('.jpg'):
                image_filepath = os.path.join(dirpath, file)
                image_paths.append(image_filepath)
                landmark_paths.append(os.path.join(landmark_path, file.replace('.jpg', '.npy')))
                diff_paths.append(os.path.join(dirpath, 'diff', file))
    assert len(image_paths) > 0, "No image found in the input directory"
    # image_paths.sort()
    # landmark_paths.sort()

    #save by au
    # for au in [4, 20, 21, 22]:
    #     os.makedirs(os.path.join(output_path, AU[au]), exist_ok=True)
    #     print(AU[au])
    #     for i in range(30):
    #         ip = image_paths[i]
    #         ld_path = landmark_paths[i]
    #         org_img = Image.open(ip).convert('RGB')
    #         ld = np.load(ld_path)
    #         ld = ld.tolist()
    #         ld = [(int(x), int(y)) for x, y in ld]
    #         h, w = org_img.size
    #         att_map, is_roi = plot_action_units_ellipsoid(au=au, h=h, w=w, lndmks=ld)
    #         # fast_draw_landmarks(img=org_img, heatmap=att_map, ldmarks=ld, wfp=os.path.join(output_path, os.path.basename(ip)))
    #         # plt.imsave(os.path.join(output_path, os.path.basename(ip)), att_map, cmap='hot')
    #         fast_draw_heatmap(img=org_img, heatmap=att_map, cl=AU[au], wfp=os.path.join(output_path, AU[au], os.path.basename(ip)))

    # save all au
    for i in range(len(image_paths)):
    # for i in range(10):
        ip = image_paths[i]
        diff_pth = diff_paths[i]
        ld_path = landmark_paths[i]
        org_img = Image.open(ip).convert('RGB')
        diff = Image.open(diff_pth).convert('RGB')
        ld = np.load(ld_path)
        ld = ld.tolist()
        ld = [(int(x), int(y)) for x, y in ld]
        h, w = org_img.size
        au_ls = []
        whole_face, _ = plot_action_units_ellipsoid(au=22, h=h, w=w, lndmks=ld)
        for au in [4, 20, 21]:
            att_map, is_roi = plot_action_units_ellipsoid(au=au, h=h, w=w, lndmks=ld)
            au_ls.append(att_map)
        union_au_heatmap = np.maximum.reduce(au_ls)
        with_face_au_heatmap = np.minimum.reduce([union_au_heatmap, whole_face])
            # plt.imsave(os.path.join(output_path, os.path.basename(ip)), att_map, cmap='hot')
        fast_draw_heatmap(img=org_img, diff=diff, heatmap=with_face_au_heatmap, cl="mask", wfp=os.path.join(output_path, os.path.basename(ip)))