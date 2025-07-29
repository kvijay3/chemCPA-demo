# Migration Guide: SAM to Cellpose-SAM

This guide outlines the migration from Facebook's Segment Anything Model (SAM) to Cellpose-SAM for cellular segmentation.

## Key Changes

### 1. Dependencies
**Before (SAM):**
```python
"segment-anything @ git+https://github.com/facebookresearch/segment-anything.git"
```

**After (Cellpose-SAM):**
```python
"cellpose"  # Includes Cellpose-SAM functionality
"scipy"
"scikit-image"
"tifffile"
"natsort"
```

### 2. Model Types
**Before (SAM):**
- `vit_b` - ViT-Base model
- `vit_l` - ViT-Large model  
- `vit_h` - ViT-Huge model

**After (Cellpose-SAM):**
- `cyto` - Original cytoplasm model
- `cyto2` - Updated cytoplasm model
- `cyto3` - Latest cytoplasm model with better generalization
- `nuclei` - Nucleus segmentation model
- `tissuenet` - Tissue segmentation model
- `livecell` - Live cell imaging model
- `cyto2_omni` - Omni model for various cell types
- And more specialized models...

### 3. Function Parameters
**Before (SAM):**
```python
def generate_sam_masks(image_s3_key, aws_config, call_id, image_db_id, sam_model_type, max_masks):
```

**After (Cellpose-SAM):**
```python
def generate_cellpose_masks(image_s3_key, aws_config, call_id, image_db_id, cellpose_model_type, max_masks, diameter=None, flow_threshold=0.4, cellprob_threshold=0.0):
```

### 4. Model Initialization
**Before (SAM):**
```python
from segment_anything import SamPredictor, sam_model_registry, SamAutomaticMaskGenerator

# Load model checkpoint
sam = sam_model_registry[sam_model_type](checkpoint=model_path)
sam.to(device=device)

# Generate masks
mask_generator = SamAutomaticMaskGenerator(
    model=sam,
    points_per_side=32,
    pred_iou_thresh=0.9,
    stability_score_thresh=0.95,
    crop_n_layers=1,
    crop_n_points_downscale_factor=2,
    min_mask_region_area=100
)
masks_data = mask_generator.generate(image_np)
```

**After (Cellpose-SAM):**
```python
from cellpose import models

# Initialize model (automatically handles model downloading/caching)
model = models.Cellpose(gpu=(device == "cuda"), model_type=cellpose_model_type)

# Generate masks
masks, flows, styles, diams = model.eval(
    image_np, 
    diameter=diameter,
    channels=[0,0],  # Grayscale or RGB
    flow_threshold=flow_threshold,
    cellprob_threshold=cellprob_threshold,
    do_3D=False
)
```

### 5. Output Processing
**Before (SAM):**
- SAM returns a list of dictionaries with segmentation masks and metadata
- Each mask has `stability_score` and `predicted_iou`

**After (Cellpose-SAM):**
- Cellpose returns a single mask array where each cell has a unique integer ID
- No individual quality scores, but generally higher quality for cellular images
- Need to extract individual masks using `np.unique(masks)`

### 6. Caching Mechanism
**Before (SAM):**
```python
# Manual checkpoint download and caching
model_path = f"/models/sam_{sam_model_type}.pth"
if not os.path.exists(model_path):
    checkpoint_url = get_sam_checkpoint_url(sam_model_type)
    os.system(f"wget -O {model_path} {checkpoint_url}")
```

**After (Cellpose-SAM):**
```python
# Automatic model management through environment variable
model_cache_dir = "/models/cellpose_models"
os.makedirs(model_cache_dir, exist_ok=True)
os.environ['CELLPOSE_LOCAL_MODELS_PATH'] = model_cache_dir
```

## Advantages of Cellpose-SAM

1. **Specialized for Cells**: Designed specifically for cellular segmentation with superior performance on biological images
2. **Better Generalization**: Works well across different cell types, imaging modalities, and experimental conditions
3. **Automatic Diameter Estimation**: Can automatically estimate cell sizes
4. **No Manual Prompting**: Unlike SAM, doesn't require points or bounding boxes
5. **Proven Track Record**: Widely used in biological research with extensive validation

## Backend Integration Changes

### API Changes
Update your backend calls from:
```python
# Old SAM call
result = generate_sam_masks.remote(
    image_s3_key=image_key,
    aws_config=aws_config,
    call_id=call_id,
    image_db_id=image_id,
    sam_model_type="vit_b",  # or vit_l, vit_h
    max_masks=50
)
```

To:
```python
# New Cellpose call
result = generate_cellpose_masks.remote(
    image_s3_key=image_key,
    aws_config=aws_config,
    call_id=call_id,
    image_db_id=image_id,
    cellpose_model_type="cyto3",  # or cyto, cyto2, nuclei, etc.
    max_masks=50,
    diameter=None,  # Auto-estimate
    flow_threshold=0.4,
    cellprob_threshold=0.0
)
```

### Model Selection Guidelines
- **`cyto3`**: Best general-purpose model for most cell types (recommended default)
- **`cyto2`**: Good alternative if cyto3 doesn't work well
- **`nuclei`**: Specifically for nucleus segmentation
- **`livecell`**: For live cell imaging
- **`tissuenet`**: For tissue-level segmentation

### Parameter Tuning
- **`diameter`**: Set to expected cell diameter in pixels, or `None` for auto-estimation
- **`flow_threshold`**: Lower values = more permissive (more masks), higher = more strict
- **`cellprob_threshold`**: Minimum cell probability (0.0 = include all detected objects)

## Testing and Validation

1. **Compare Results**: Run both SAM and Cellpose-SAM on the same images to compare quality
2. **Performance Testing**: Monitor processing time and resource usage
3. **Quality Assessment**: Evaluate segmentation accuracy on your specific cell types
4. **Parameter Optimization**: Tune Cellpose parameters for your specific use case

## Deployment Steps

1. **Update Modal App**: Deploy the new `cellpose_sam_processor.py`
2. **Update Backend**: Modify API calls to use new function and parameters
3. **Update Frontend**: Ensure UI can handle new model types and parameters
4. **Test Thoroughly**: Validate end-to-end functionality
5. **Monitor Performance**: Watch for any issues in production

## Rollback Plan

Keep the original SAM implementation available during the transition period. You can run both systems in parallel and gradually migrate users to Cellpose-SAM after validation.

