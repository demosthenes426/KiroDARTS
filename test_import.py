import sys
sys.path.append('src')

try:
    from model_factory import ModelFactory
    print("ModelFactory imported successfully")
    factory = ModelFactory()
    print("ModelFactory instantiated successfully")
except Exception as e:
    print(f"Error: {e}")
    import traceback
    traceback.print_exc()