from clingon import clingon
import cv2

# sources:
# https://docs.opencv.org/3.1.0/d5/daf/tutorial_py_histogram_equalization.html
# https://stackoverflow.com/questions/25008458/how-to-apply-clahe-on-rgb-color-images


def clahe_rgb(img, clahe):
    lab = cv2.cvtColor(img, cv2.COLOR_BGR2LAB)
    lab_planes = cv2.split(lab)
    lab_planes[0] = clahe.apply(lab_planes[0])
    lab = cv2.merge(lab_planes)
    return cv2.cvtColor(lab, cv2.COLOR_LAB2BGR)


@clingon.clize()
def runner(image_path, black_and_white=False, overwrite=False):
    """ Contrast Limited Adaptive Histogram Equalization
    based on OpenCV
    """
    img = cv2.imread(image_path, 1-int(black_and_white))
    clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8))
    if black_and_white:
        img = clahe.apply(img)
    else:
        img = clahe_rgb(img, clahe)
    if overwrite:
        target = image_path
    else:
        base, ext = image_path.split('.')
        target = '.'.join((base + '_', ext))
    cv2.imwrite(target, img)
