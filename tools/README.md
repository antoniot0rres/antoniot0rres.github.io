# Scene builder

`build_scene.py` turns a photo into the layered scene that `index.html` renders.

## Changing the photo

```bash
pip install numpy opencv-contrib-python-headless scikit-image scipy onnxruntime
python3 tools/build_scene.py path/to/photo.jpg
```

This overwrites `scene/`. Commit the folder and the site picks it up. The first run downloads the depth model (~390 MB) to `~/.cache/depth-anything/`; a run takes about one minute on a laptop CPU.

Landscape photos with clear foreground, middle ground and background give the strongest effect.

## How it works

The previous version displaced every pixel by a continuous depth map. Wherever depth varied along a straight edge, the edge bent, and at depth steps the hidden background was invented by stretching edge pixels.

The scene is now a set of **planar regions**:

1. Relative depth from Depth Anything (ViT-B, ONNX, CPU).
2. Depth edges snapped to the photo's edges with an image-guided weighted median, so no in-between slivers survive.
3. Superpixels on colour plus depth, each fitted with a plane, merged greedily while the merged plane still fits. A slope prior keeps thin strips fronto-parallel.
4. Region borders re-decided per pixel with a guided filter; pixels on a strong depth step go to the nearer object.
5. Surfaces hidden behind nearer regions are filled by mirroring the region's own texture, only as far as the largest possible shift can reveal. Up to three hidden surfaces are kept per pixel.

In the browser, each region is drawn as a rigid plane. A plane under a viewpoint shift moves by an affine map, and affine maps keep straight lines straight, so nothing bends. A depth buffer lets the nearest surface win at every pixel, and a second pass closes sub-pixel seams between neighbouring planes.

## Files in `scene/`

| File | Content |
|---|---|
| `image.jpg` | The photo, 2400 px wide |
| `labels.png` | R: region id per pixel. G, B: 1st and 2nd hidden surface |
| `labels2.png` | R: 3rd hidden surface |
| `fill1.jpg` to `fill3.jpg` | Colour of the hidden surfaces |
| `layers.json` | Per region: plane `depth = a*u + b*v + c` and bounding box, in uv |

## Limits

- The fill supports `strength` up to 0.045 (the page uses 0.034). Above that, rebuild with a higher `STRENGTH_MAX`.
- Very thin structures over a distant background (cords, wires) can show a faint duplicate at the extreme mouse positions.
