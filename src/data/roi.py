from PIL import Image

ROI_X_START = 0.25
ROI_X_END   = 0.75
ROI_Y_START = 0.20
ROI_Y_END   = 0.80


def crop_canine_roi(img: Image.Image) -> Image.Image:
    w, h = img.size
    left   = int(w * ROI_X_START)
    right  = int(w * ROI_X_END)
    top    = int(h * ROI_Y_START)
    bottom = int(h * ROI_Y_END)
    return img.crop((left, top, right, bottom))
