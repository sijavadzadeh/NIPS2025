import os
import torch
import warnings

def get_data_directory():
    username = os.getlogin()
    if username == "Maral":
        return "D:\\Sina\\Data\\period9\\"
    elif username == "Lab User":
        return "F:\\Python Projects\\data\\period9\\"
    elif username == "Rahil":
        return "C:\\Sina\\Data\\period9\\"
    else:
        raise ValueError(f"Unknown user: {username}")
    

def get_device():
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    if torch.cuda.is_available() == False:
        warnings.warn("⚠️ CUDA is not available. The model will run on CPU, which may be slower.", RuntimeWarning)
    return device