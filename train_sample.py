import h5py
import json

def remove_quantization_config(config_data):
    """Recursively searches for and deletes the 'quantization_config' key."""
    if isinstance(config_data, dict):
        config_data.pop('quantization_config', None)
        for key, value in config_data.items():
            remove_quantization_config(value)
    elif isinstance(config_data, list):
        for item in config_data:
            remove_quantization_config(item)

# Open the H5 file in read/write mode
file_path = 'model/optimized_model.h5'

try:
    print(f"Opening {file_path} to patch the config...")
    with h5py.File(file_path, 'r+') as f:
        if 'model_config' in f.attrs:
            # Load the internal configuration
            model_config = json.loads(f.attrs['model_config'])
            
            # Delete the problematic setting
            remove_quantization_config(model_config)
            
            # Save it back to the file
            f.attrs['model_config'] = json.dumps(model_config).encode('utf-8')
            print("✅ Successfully patched! The model is now compatible with Windows.")
        else:
            print("⚠️ No model_config found in the file.")
except Exception as e:
    print(f"❌ Error: {e}")