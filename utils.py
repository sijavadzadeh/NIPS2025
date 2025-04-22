import os

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