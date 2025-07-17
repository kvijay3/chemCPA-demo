import modal

app = modal.App("chemcpa-fix")
volume = modal.Volume.from_name("chemcpa-data")

@app.function(volumes={"/data": volume})
def fix_lightning_module():
    """Quick fix for the val_loss issue"""
    from pathlib import Path
    
    data_dir = Path("/data")
    lightning_module_path = data_dir / "chemCPA" / "lightning_module.py"
    
    # Check if the file exists
    if not lightning_module_path.exists():
        print(f"❌ File not found: {lightning_module_path}")
        print("📁 Available files in /data:")
        for item in data_dir.rglob("*"):
            if item.is_file():
                print(f"   {item}")
        return {"error": "lightning_module.py not found"}
    
    # Read the current file
    with open(lightning_module_path, 'r') as f:
        content = f.read()
    
    # Add the val_loss logging fix
    old_code = '''        # Print validation results
        print(f"Validation R² scores: Mean={result[0]:.4f}, Mean_DE={result[1]:.4f}, Var={result[2]:.4f}, Var_DE={result[3]:.4f}")
        self.train()'''
    
    new_code = '''        # Log val_loss for ModelCheckpoint monitoring (use negative R2_mean as loss)
        val_loss = -result[0]  # Negative R2_mean (higher R2 = lower loss)
        self.log("val_loss", val_loss, on_step=False, on_epoch=True, prog_bar=True)
        
        # Print validation results
        print(f"Validation R² scores: Mean={result[0]:.4f}, Mean_DE={result[1]:.4f}, Var={result[2]:.4f}, Var_DE={result[3]:.4f}")
        self.train()'''
    
    if old_code in content:
        content = content.replace(old_code, new_code)
        
        # Write back the fixed file
        with open(lightning_module_path, 'w') as f:
            f.write(content)
        
        print("✅ Fixed val_loss logging in lightning_module.py")
        return {"status": "fixed"}
    else:
        print("⚠️ Could not find the code to replace")
        return {"status": "not_found"}

if __name__ == "__main__":
    with app.run():
        result = fix_lightning_module.remote()
        print("Fix result:", result)
