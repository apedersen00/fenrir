import sys
import os
import importlib.util

# Check Python version
print("=" * 50)
print("🔍 Checking Python Environment")
print("=" * 50)
print(f"Python Version: {sys.version}")
print(f"Python Executable: {sys.executable}")

# Check if running inside a Conda environment
conda_prefix = os.environ.get("CONDA_PREFIX")
if conda_prefix:
    print(f"✅ Running inside Conda environment: {conda_prefix}")
else:
    print("⚠️ WARNING: Not running inside a Conda environment!")

# Check CUDA availability
try:
    import torch
    cuda_available = torch.cuda.is_available()
    device_count = torch.cuda.device_count()
    print("\n🔥 CUDA Availability Check")
    print("-" * 30)
    if cuda_available:
        print(f"✅ CUDA is available. {device_count} GPU(s) detected.")
        for i in range(device_count):
            print(f"  - GPU {i}: {torch.cuda.get_device_name(i)}")
    else:
        print("⚠️ WARNING: CUDA is NOT available. Using CPU instead.")
except ImportError:
    print("⚠️ WARNING: PyTorch is not installed. Cannot check CUDA availability.")

# Required packages and their expected versions (set None if version doesn't matter)
required_packages = {
    "snntorch": None, 
    "torch": None,
    "torchvision": None,
    "torchaudio": None,
    "numpy": None,
    "jupyter": None,
    "matplotlib": None
}

print("\n📦 Checking Required Packages")
print("=" * 50)

# Function to check if package is installed and print version
def check_package(package_name, expected_version=None):
    try:
        spec = importlib.util.find_spec(package_name)
        if spec is None:
            print(f"❌ {package_name} is NOT installed.")
        else:
            module = importlib.import_module(package_name)
            installed_version = getattr(module, "__version__", "Unknown")
            if expected_version and installed_version != expected_version:
                print(f"⚠️ {package_name} version mismatch: Installed {installed_version}, Expected {expected_version}")
            else:
                print(f"✅ {package_name} is installed (Version: {installed_version})")
    except Exception as e:
        print(f"❌ Error checking {package_name}: {e}")

# Check each required package
for package, version in required_packages.items():
    check_package(package, version)

print("\n✅ Environment check completed.")
