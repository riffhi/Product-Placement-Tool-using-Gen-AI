# import torch
# # print(torch.cuda.is_available())  # Should return True
# print(torch.version.cuda)         # Should return 12.6

# if torch.cuda.is_available():
#     print(f"Using GPU: {torch.cuda.get_device_name(0)}")
# else:
#     print("Using CPU")
import triton
print(triton.__version__)  # Should return the installed version