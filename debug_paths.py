import os
import sys
import importlib

# Print current working directory
print(f"Current working directory: {os.getcwd()}")

# List files in current directory
print(f"Files in current directory: {os.listdir('.')}")

# Check if modules directory exists
if os.path.exists('modules'):
    print(f"Files in modules directory: {os.listdir('modules')}")
else:
    print("modules directory not found")

# Try to import the module to check what's actually being loaded
try:
    import modules.attention
    print(f"Successfully imported modules.attention from: {modules.attention.__file__}")
    
    # Try to inspect the module content
    with open(modules.attention.__file__, 'r') as f:
        content = f.read()
        print(f"First 100 characters of file content: {content[:100]}")
        
        # Check if getattr is in the file
        if "getattr" in content:
            print("getattr found in file content")
        else:
            print("getattr NOT found in file content")
            
except Exception as e:
    print(f"Error importing modules.attention: {e}")