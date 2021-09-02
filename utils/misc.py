# import yaml
import time
import math
import sys


# Timestamp
def getTimestamp():
    """
    Get timestamp.
    :return: string, with format like '2019_04_02_123030',
             i.e. <year>_<mon>_<day>_<hour><min><sec>
    """
    return time.strftime("%Y_%m_%d_%H%M%S", time.localtime())

# Print log
def print_log_Fn_Builder(log_name):
    def print_log(s, end=''):
        print(s, end=end)
        with open(log_name, 'a') as f:
            f.write(s)
    return print_log


# Timer
def asMinutes(s):
    m = math.floor(s / 60)
    s -= m * 60
    return '%dm %ds' % (m, s)

def asHours(s):
    h = math.floor(s / 60 / 60)
    s -= h * 60 * 60
    m = math.floor(s / 60)
    s -= m * 60
    return '%dh %dm %ds' % (h, m, s)

def timeSince(since, percent):
    now = time.time()
    s = now - since
    es = s / (percent)
    rs = es - s
    return '%s (- %s)' % (asMinutes(s) if s < 3600 else asHours(s),
                          asMinutes(rs) if rs < 3600 else asHours(rs))

def timeBetween(_from, _to):
    s = _to - _from
    if s > 3600:
        return asHours(s)
    else:
        return asMinutes(s)
    
def removeFromList(l, rm_indexes):
    rm_indexes.sort()
    count = 0
    for ind in rm_indexes:
        l.pop(ind-count)
        count += 1
    return l

class Logger(object):
  def __init__(self, output_file):
    self.terminal = sys.stdout
    self.log = open(output_file, "a")

  def write(self, message):
    print(message, end="", file=self.terminal, flush=True)
    print(message, end="", file=self.log, flush=True)

  def flush(self):
    self.terminal.flush()
    self.log.flush()

