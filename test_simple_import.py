import sys
sys.path.append('src')

# Test if the file can be imported at all
try:
    import model_factory
    print("Module imported successfully")
    print(f"Module contents: {dir(model_factory)}")
except Exception as e:
    print(f"Module import error: {e}")
    import traceback
    traceback.print_exc()