"""Pure image geometry of the two new inputs. No file, label or model is read here."""
import numpy as np
from PIL import Image

OUTPUT = 224
A08_SHORT_SIDE = 256


def a08_side(width, height):
    """How many source pixels of the short side A08's crop keeps (short side 256, centre 224)."""
    return OUTPUT * min(width, height) / A08_SHORT_SIDE


def a08_centre(width, height):
    """The source-pixel centre for which zoom_crop reproduces A08's centre crop."""
    scale = OUTPUT / a08_side(width, height)
    resized_w, resized_h = max(OUTPUT, round(width * scale)), max(OUTPUT, round(height * scale))
    return ((resized_w - OUTPUT) // 2 + OUTPUT // 2) / scale, ((resized_h - OUTPUT) // 2 + OUTPUT // 2) / scale


def zoom_crop(image, centre_x, centre_y, side, size=OUTPUT):
    """Resize the whole frame so that `side` source pixels become `size`, then take a zero-padded square about the centre."""
    if side <= 0:
        raise ValueError('a positive crop side is required')
    height, width = image.shape[:2]
    scale = size / side
    resized_w, resized_h = max(1, round(width * scale)), max(1, round(height * scale))
    resized = np.asarray(Image.fromarray(image).resize((resized_w, resized_h), Image.BILINEAR), np.uint8)
    left, top = round(centre_x * scale) - size // 2, round(centre_y * scale) - size // 2
    out = np.zeros((size, size, 3), np.uint8)
    x0, y0, x1, y1 = max(0, left), max(0, top), min(resized_w, left + size), min(resized_h, top + size)
    if x1 > x0 and y1 > y0:
        out[y0 - top:y1 - top, x0 - left:x1 - left] = resized[y0:y1, x0:x1]
    return out


def side_ratio(a08_source_side, helena_sides):
    """Helena's crop side to A08's magnification. Computed on EV9V only."""
    sides = np.asarray(helena_sides, float)
    if not len(sides) or (sides <= 0).any():
        raise ValueError('positive Helena crop sides are required')
    return float(a08_source_side / np.median(sides))


def pack_mask(mask):
    mask = np.asarray(mask, bool)
    return {'shape': tuple(int(x) for x in mask.shape), 'bits': np.packbits(mask.ravel()).tobytes()}


def unpack_mask(packed):
    height, width = packed['shape']
    bits = np.unpackbits(np.frombuffer(packed['bits'], np.uint8), count=height * width)
    return bits.reshape(height, width).astype(bool)


def segment_at(segments, frame):
    """The layout segment holding this source frame; a frame past the last segment belongs to the last."""
    if not segments:
        raise ValueError('at least one layout segment is required')
    for segment in segments:
        if segment['start'] <= frame < segment['end']:
            return segment
    if frame >= segments[-1]['end']:
        return segments[-1]
    raise ValueError(f'frame {frame} lies in no layout segment')


def apply_mask(image, mask):
    if image.shape[:2] != mask.shape:
        raise ValueError('the mask and the frame differ in size')
    return image * mask[:, :, None].astype(np.uint8)


def z_frame(image, segments, centre_x, centre_y, side, frame):
    """Arm Z: mask of the frame's layout segment, then a crop at the given side about Helena's centre."""
    return zoom_crop(apply_mask(image, unpack_mask(segment_at(segments, frame)['mask'])), centre_x, centre_y, side)


def full_frame_square(image, size=OUTPUT):
    """Fallback of arm H when no geometry exists: the whole frame padded to a square, without a mask."""
    height, width = image.shape[:2]
    side = max(height, width)
    canvas = np.zeros((side, side, 3), np.uint8)
    top, left = (side - height) // 2, (side - width) // 2
    canvas[top:top + height, left:left + width] = image
    return np.asarray(Image.fromarray(canvas).resize((size, size), Image.BILINEAR), np.uint8)
