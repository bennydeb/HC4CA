
from datetime import datetime

def get_timestamp():
    time = datetime.now().strftime("%H%M%S%d%m%y")
    return time
