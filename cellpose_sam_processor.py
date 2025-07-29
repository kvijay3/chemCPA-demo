import modal

# Modal imports will be done inside functions to avoid local dependency issues

# Define the Modal app
app = modal.App("cellpose-sam-segmentation-processor")

# Create a volume for caching Cellpose models
model_volume = modal.Volume.from_name("cellpose-models", create_if_missing=True)

# Create the image with Cellpose-SAM dependencies
image = modal.Image.debian_slim().apt_install(
    "git", "libglib2.0-0", "libsm6", "libxext6", "libxrender-dev", "libgomp1", "wget"
).pip_install(
    "torch",
    "torchvision", 
    "opencv-python-headless",
    "numpy",
    "boto3",
    "Pillow",
    "matplotlib",
    "scipy",
    "scikit-image",
    "tifffile",
    "natsort",
    "cellpose"  # Main Cellpose package which includes Cellpose-SAM
)

# Generate Cellpose-SAM masks
@app.function(
    image=image,
    gpu="T4",
    memory=8192,
    timeout=900,  # 15 minutes timeout
    volumes={"/models": model_volume},  # Mount volume for model caching
    secrets=[modal.Secret.from_name("aws-credentials")]
)
def generate_cellpose_masks(image_s3_key, aws_config, call_id, image_db_id, cellpose_model_type, max_masks, diameter=None, flow_threshold=0.4, cellprob_threshold=0.0):
    """
    Generate Cellpose-SAM masks for an image and return results for Django to process
    
    Args:
        image_s3_key: S3 key of the image
        aws_config: AWS configuration
        call_id: Django tracking ID for this operation
        image_db_id: Django Image model ID
        cellpose_model_type: Cellpose model type ('cyto', 'nuclei', 'cyto2', 'cyto3', etc.)
        max_masks: Maximum number of masks to return (None for unlimited)
        diameter: Expected diameter of objects (None for auto-estimation)
        flow_threshold: Flow error threshold for masks
        cellprob_threshold: Cell probability threshold
    
    Returns:
        dict: Success status and mask data for Django to store
    """
    # Import everything inside the function to avoid local dependency issues
    import numpy as np
    import boto3
    from io import BytesIO
    import os
    import cv2
    from PIL import Image
    import json
    import logging
    import torch
    from cellpose import models, io
    
    # Configure logging
    logging.basicConfig(level=logging.INFO)
    logger = logging.getLogger(__name__)
    
    try:
        logger.info(f"Starting Cellpose-SAM processing for call_id: {call_id}, image_id: {image_db_id}")
        
        # Check GPU availability
        device = "cuda" if torch.cuda.is_available() else "cpu"
        logger.info(f"Using device: {device}")
        
        # Download image from S3
        logger.info("Downloading image from S3...")
        s3_client = boto3.client(
            's3',
            aws_access_key_id=aws_config['access_key'],
            aws_secret_access_key=aws_config['secret_key'],
            region_name=aws_config['region']
        )
        
        # Download image
        image_obj = s3_client.get_object(Bucket=aws_config['bucket'], Key=image_s3_key)
        image_data = image_obj['Body'].read()
        
        # Load image
        image = Image.open(BytesIO(image_data)).convert('RGB')
        image_np = np.array(image)
        logger.info(f"Image loaded with shape: {image_np.shape}")
        
        # Set up model cache directory
        model_cache_dir = "/models/cellpose_models"
        os.makedirs(model_cache_dir, exist_ok=True)
        os.environ['CELLPOSE_LOCAL_MODELS_PATH'] = model_cache_dir
        
        # Load Cellpose model
        logger.info(f"Loading Cellpose model: {cellpose_model_type}")
        
        # Initialize the model with caching
        if cellpose_model_type in ['cyto', 'cyto2', 'cyto3']:
            model = models.Cellpose(gpu=(device == "cuda"), model_type=cellpose_model_type)
        elif cellpose_model_type == 'nuclei':
            model = models.Cellpose(gpu=(device == "cuda"), model_type='nuclei')
        else:
            # Default to cyto if unknown model type
            logger.warning(f"Unknown model type {cellpose_model_type}, defaulting to 'cyto'")
            model = models.Cellpose(gpu=(device == "cuda"), model_type='cyto')
        
        # Persist any downloaded models to the volume
        model_volume.commit()
        
        # Estimate diameter if not provided
        if diameter is None:
            logger.info("Estimating cell diameter...")
            diameter = model.sz.eval(image_np, channels=[0,0])[0]
            logger.info(f"Estimated diameter: {diameter}")
        
        # Generate masks using Cellpose
        logger.info("Generating masks with Cellpose-SAM...")
        masks, flows, styles, diams = model.eval(
            image_np, 
            diameter=diameter,
            channels=[0,0],  # Grayscale or RGB
            flow_threshold=flow_threshold,
            cellprob_threshold=cellprob_threshold,
            do_3D=False
        )
        
        # Get unique mask IDs (excluding background which is 0)
        unique_masks = np.unique(masks)
        unique_masks = unique_masks[unique_masks > 0]  # Remove background
        
        logger.info(f"Cellpose generated {len(unique_masks)} masks")
        
        # Process and filter masks
        results = []
        unlimited_mode = (max_masks is None)
        filtered_count = 0
        local_mask_index = 1  # Start from 1 for each image
        
        for mask_id in unique_masks:
            try:
                # Create binary mask for this cell
                binary_mask = (masks == mask_id).astype(np.uint8)
                
                # Process the mask
                mask_info = process_cellpose_mask(
                    binary_mask, 
                    local_mask_index, 
                    mask_id, 
                    unlimited_mode=unlimited_mode
                )
                
                if mask_info:
                    # Add Cellpose-specific metadata
                    mask_info.update({
                        'cellpose_score': float(1.0),  # Cellpose doesn't provide individual scores like SAM
                        'diameter_used': float(diameter),
                        'flow_threshold': float(flow_threshold),
                        'cellprob_threshold': float(cellprob_threshold)
                    })
                    results.append(mask_info)
                    local_mask_index += 1  # Increment local counter only for successful masks
                else:
                    filtered_count += 1
            except Exception as e:
                logger.warning(f"Failed to process mask {mask_id}: {str(e)}")
                filtered_count += 1
                continue
        
        logger.info(f"Filtered out {filtered_count} masks during processing (unlimited_mode: {unlimited_mode})")
        
        # Sort by area (larger cells first) since Cellpose doesn't provide quality scores like SAM
        results.sort(key=lambda x: x.get('area_pixels', 0), reverse=True)
        
        # Debug logging
        logger.info(f"max_masks parameter: {max_masks}")
        logger.info(f"Total results before limiting: {len(results)}")
        
        # Limit results only if max_masks is specified (not None)
        if max_masks is not None:
            results = results[:max_masks]  # User-specified number of masks
            logger.info(f"Limited to {max_masks} masks")
        else:
            logger.info("Unlimited mode - keeping all masks")
        
        logger.info(f"Successfully processed {len(results)} high-quality masks")
        
        # Store results in S3 for Django to retrieve
        try:
            results_key = f"cellpose_results/{call_id}.json"
            results_data = {
                'success': True,
                'masks': results,
                'total_generated': len(unique_masks),
                'filtered_count': len(results),
                'call_id': call_id,
                'image_db_id': str(image_db_id),  # Convert UUID to string
                'cellpose_model': cellpose_model_type,  # Include the Cellpose model used
                'diameter': float(diameter),
                'completed_at': str(image.size)  # Use a timestamp-like field
            }
            
            s3_client.put_object(
                Bucket=aws_config['bucket'],
                Key=results_key,
                Body=json.dumps(results_data),
                ContentType='application/json'
            )
            logger.info(f"Stored Cellpose results in S3: {results_key}")
            
        except Exception as s3_error:
            logger.warning(f"Failed to store results in S3: {str(s3_error)}")
        
        return {
            'success': True,
            'masks': results,
            'total_generated': len(unique_masks),
            'filtered_count': len(results),
            'call_id': call_id,
            'image_db_id': image_db_id,
            'diameter': diameter
        }
        
    except Exception as e:
        logger.error(f"Error in Cellpose segmentation: {str(e)}")
        
        return {
            'success': False,
            'error': str(e),
            'call_id': call_id,
            'image_db_id': image_db_id
        }


def process_cellpose_mask(mask, index, mask_id, unlimited_mode=False):
    """Process a Cellpose mask to extract features and boundaries"""
    try:
        import cv2
        import numpy as np
        
        # Find contours
        mask_uint8 = (mask * 255).astype(np.uint8)
        contours, _ = cv2.findContours(mask_uint8, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        
        if not contours:
            return None
        
        # Get the largest contour
        largest_contour = max(contours, key=cv2.contourArea)
        
        # Calculate area and perimeter
        area = cv2.contourArea(largest_contour)
        perimeter = cv2.arcLength(largest_contour, True)
        
        # Skip very small objects only in limited mode
        if not unlimited_mode and area < 100:
            return None
        
        # Calculate bounding rectangle for basic aspect ratio
        x, y, w, h = cv2.boundingRect(largest_contour)
        basic_aspect_ratio = float(w) / h if h > 0 else 0
        
        # Calculate PCA-based aspect ratio for more accurate measurement
        points = largest_contour.reshape(-1, 2)
        pca_aspect_ratio = calculate_pca_aspect_ratio_from_points(points)
        
        # Simplify boundary for storage (reduce number of points)
        epsilon = 0.002 * cv2.arcLength(largest_contour, True)
        simplified_contour = cv2.approxPolyDP(largest_contour, epsilon, True)
        boundary_points = simplified_contour.reshape(-1, 2).tolist()
        
        return {
            'name': f'Cell {index}',  # Changed from 'Mask' to 'Cell' for Cellpose
            'aspect_ratio': pca_aspect_ratio,
            'basic_aspect_ratio': basic_aspect_ratio,
            'area_pixels': int(area),
            'perimeter_pixels': float(perimeter),
            'mask_id': int(mask_id),  # Include the original mask ID from Cellpose
            'boundaries': {
                'points': boundary_points,
                'min_x': float(x),
                'max_x': float(x + w),
                'min_y': float(y),
                'max_y': float(y + h),
                'width': float(w),
                'height': float(h)
            }
        }
        
    except Exception as e:
        import logging
        logger = logging.getLogger(__name__)
        logger.error(f"Error processing Cellpose mask {index}: {str(e)}")
        return None


def calculate_pca_aspect_ratio_from_points(points):
    """Calculate PCA-based aspect ratio from contour points"""
    try:
        import numpy as np
        if len(points) < 3:
            return 1.0
        
        # Center the points
        centered_points = points - np.mean(points, axis=0)
        
        # Calculate covariance matrix
        cov_matrix = np.cov(centered_points.T)
        
        # Get eigenvalues
        eigenvalues = np.linalg.eigvals(cov_matrix)
        eigenvalues = np.sort(eigenvalues)[::-1]
        
        # Calculate aspect ratio
        if eigenvalues[1] > 0:
            aspect_ratio = np.sqrt(eigenvalues[0] / eigenvalues[1])
        else:
            aspect_ratio = 1.0
            
        return float(aspect_ratio)
        
    except Exception:
        return 1.0


def upload_mask_to_s3(mask, s3_key, s3_client, bucket_name, aws_config):
    """Upload mask image to S3"""
    try:
        import numpy as np
        from PIL import Image
        from io import BytesIO
        import logging
        
        logger = logging.getLogger(__name__)
        # Convert mask to image
        mask_pil = Image.fromarray((mask * 255).astype(np.uint8))
        mask_buffer = BytesIO()
        mask_pil.save(mask_buffer, format='PNG')
        mask_buffer.seek(0)
        
        # Upload to S3
        s3_client.put_object(
            Bucket=bucket_name,
            Key=s3_key,
            Body=mask_buffer.getvalue(),
            ContentType='image/png'
        )
        
        # Return S3 URL (will be replaced with presigned URL in backend)
        return f"https://{bucket_name}.s3.{aws_config.get('region')}.amazonaws.com/{s3_key}"
        
    except Exception as e:
        logger.error(f"Failed to upload mask to S3: {str(e)}")
        return None


# Additional helper function for model type validation
def get_available_cellpose_models():
    """Get list of available Cellpose model types"""
    return [
        'cyto',      # Original cytoplasm model
        'cyto2',     # Updated cytoplasm model
        'cyto3',     # Latest cytoplasm model with better generalization
        'nuclei',    # Nucleus segmentation model
        'tissuenet', # Tissue segmentation model
        'livecell',  # Live cell imaging model
        'cyto2_omni', # Omni model for various cell types
        'bact_phase_omni', # Bacterial phase contrast
        'bact_fluor_omni', # Bacterial fluorescence
        'deepbacs',  # Deep learning bacterial segmentation
        'cyto2_cp3', # CellPose 3.0 cytoplasm model
    ]


# Function to validate model type
def validate_cellpose_model_type(model_type):
    """Validate if the provided model type is supported"""
    available_models = get_available_cellpose_models()
    if model_type not in available_models:
        return False, f"Model type '{model_type}' not supported. Available models: {', '.join(available_models)}"
    return True, "Valid model type"


if __name__ == "__main__":
    # Example usage for testing
    print("Cellpose-SAM Modal processor loaded successfully!")
    print("Available model types:", get_available_cellpose_models())

