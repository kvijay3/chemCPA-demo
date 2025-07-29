#!/usr/bin/env python3
"""
Example usage of the Cellpose-SAM Modal processor

This script demonstrates how to call the Cellpose-SAM segmentation function
and handle the results.
"""

import modal
from cellpose_sam_processor import app

def example_usage():
    """Example of how to use the Cellpose-SAM processor"""
    
    # Example parameters (replace with your actual values)
    image_s3_key = "path/to/your/image.jpg"
    aws_config = {
        'access_key': 'your-access-key',
        'secret_key': 'your-secret-key',
        'region': 'us-east-1',
        'bucket': 'your-bucket-name'
    }
    call_id = "unique-call-id-123"
    image_db_id = "image-database-id-456"
    
    # Cellpose-SAM specific parameters
    cellpose_model_type = "cyto3"  # Best general-purpose model
    max_masks = 50  # Limit number of returned masks
    diameter = None  # Auto-estimate cell diameter
    flow_threshold = 0.4  # Default flow threshold
    cellprob_threshold = 0.0  # Include all detected cells
    
    # Call the Cellpose-SAM processor
    with app.run():
        result = app.generate_cellpose_masks.remote(
            image_s3_key=image_s3_key,
            aws_config=aws_config,
            call_id=call_id,
            image_db_id=image_db_id,
            cellpose_model_type=cellpose_model_type,
            max_masks=max_masks,
            diameter=diameter,
            flow_threshold=flow_threshold,
            cellprob_threshold=cellprob_threshold
        )
    
    # Process the results
    if result['success']:
        print(f"✅ Successfully processed {len(result['masks'])} masks")
        print(f"📊 Total masks generated: {result['total_generated']}")
        print(f"🎯 Diameter used: {result.get('diameter', 'N/A')}")
        
        # Example of accessing individual mask data
        for i, mask in enumerate(result['masks'][:3]):  # Show first 3 masks
            print(f"\n🔬 {mask['name']}:")
            print(f"   Area: {mask['area_pixels']} pixels")
            print(f"   Aspect ratio: {mask['aspect_ratio']:.2f}")
            print(f"   Bounding box: {mask['boundaries']['width']}x{mask['boundaries']['height']}")
            
    else:
        print(f"❌ Processing failed: {result['error']}")
    
    return result


def compare_model_types():
    """Example comparing different Cellpose model types"""
    
    model_types = ['cyto', 'cyto2', 'cyto3', 'nuclei']
    
    # Your image and AWS config here
    image_s3_key = "path/to/your/test-image.jpg"
    aws_config = {
        'access_key': 'your-access-key',
        'secret_key': 'your-secret-key', 
        'region': 'us-east-1',
        'bucket': 'your-bucket-name'
    }
    
    results = {}
    
    with app.run():
        for model_type in model_types:
            print(f"\n🧪 Testing model: {model_type}")
            
            result = app.generate_cellpose_masks.remote(
                image_s3_key=image_s3_key,
                aws_config=aws_config,
                call_id=f"test-{model_type}",
                image_db_id="test-image",
                cellpose_model_type=model_type,
                max_masks=20,
                diameter=None,
                flow_threshold=0.4,
                cellprob_threshold=0.0
            )
            
            results[model_type] = result
            
            if result['success']:
                print(f"   ✅ Found {len(result['masks'])} masks")
                avg_area = sum(m['area_pixels'] for m in result['masks']) / len(result['masks'])
                print(f"   📏 Average cell area: {avg_area:.0f} pixels")
            else:
                print(f"   ❌ Failed: {result['error']}")
    
    # Compare results
    print("\n📊 Model Comparison Summary:")
    for model_type, result in results.items():
        if result['success']:
            mask_count = len(result['masks'])
            avg_area = sum(m['area_pixels'] for m in result['masks']) / mask_count if mask_count > 0 else 0
            print(f"   {model_type:8}: {mask_count:3d} masks, avg area: {avg_area:6.0f} pixels")
        else:
            print(f"   {model_type:8}: FAILED")
    
    return results


def parameter_tuning_example():
    """Example of tuning Cellpose parameters for your specific use case"""
    
    # Test different parameter combinations
    parameter_sets = [
        {"flow_threshold": 0.2, "cellprob_threshold": 0.0, "name": "Permissive"},
        {"flow_threshold": 0.4, "cellprob_threshold": 0.0, "name": "Default"},
        {"flow_threshold": 0.6, "cellprob_threshold": 0.5, "name": "Strict"},
    ]
    
    image_s3_key = "path/to/your/test-image.jpg"
    aws_config = {
        'access_key': 'your-access-key',
        'secret_key': 'your-secret-key',
        'region': 'us-east-1', 
        'bucket': 'your-bucket-name'
    }
    
    print("🔧 Parameter Tuning Results:")
    
    with app.run():
        for params in parameter_sets:
            result = app.generate_cellpose_masks.remote(
                image_s3_key=image_s3_key,
                aws_config=aws_config,
                call_id=f"tune-{params['name'].lower()}",
                image_db_id="tune-test",
                cellpose_model_type="cyto3",
                max_masks=None,  # No limit for tuning
                diameter=None,
                flow_threshold=params["flow_threshold"],
                cellprob_threshold=params["cellprob_threshold"]
            )
            
            if result['success']:
                mask_count = len(result['masks'])
                print(f"   {params['name']:10}: {mask_count:3d} masks "
                      f"(flow={params['flow_threshold']}, prob={params['cellprob_threshold']})")
            else:
                print(f"   {params['name']:10}: FAILED - {result['error']}")


if __name__ == "__main__":
    print("🔬 Cellpose-SAM Modal Processor Examples")
    print("=" * 50)
    
    # Uncomment the example you want to run:
    
    # Basic usage example
    # example_usage()
    
    # Model comparison example  
    # compare_model_types()
    
    # Parameter tuning example
    # parameter_tuning_example()
    
    print("\n💡 Remember to:")
    print("   1. Replace example AWS config with your actual credentials")
    print("   2. Update image_s3_key with your actual image path")
    print("   3. Deploy the Modal app before running these examples")
    print("   4. Ensure your AWS credentials are properly configured in Modal secrets")

