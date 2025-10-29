import sys, os, distutils.core
import torch

def initial_detectron():
    print("Inital Detectron is called")
    TORCH_VERSION = ".".join(torch.__version__.split(".")[:2])
    CUDA_VERSION = torch.__version__.split("+")[-1]
    print("torch: ", TORCH_VERSION, "; cuda: ", CUDA_VERSION)
    print(torch.cuda.is_available())
    print(torch.cuda.device_count())

    # Note: This is a faster way to install detectron2 in Colab, but it does not include all functionalities (e.g. compiled operators).
    # See https://detectron2.readthedocs.io/tutorials/install.html for full installation instructions
    try:
        dist = distutils.core.run_setup("../detectron2/setup.py")
        # !python -m pip install {' '.join([f"'{x}'" for x in dist.install_requires])}
        sys.path.insert(0, os.path.abspath("../detectron2"))
    except Exception as e:
        print("Exception", e)
    # Properly install detectron2. (Please do not install twice in both ways)
    # !python -m pip install 'git+https://github.com/facebookresearch/detectron2.git'
