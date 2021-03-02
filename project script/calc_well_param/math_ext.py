import math

def sign(x):
    if x>0: return 1
    elif x<0: return -1
    elif x==0: return 0

def pow_(_base, _grade):
    if (_base == None or _grade == None): return None
    if (_grade // 2 == 0): val = 1
    elif (_grade < 1 and ((1 / _grade) // 2 == 0) and (_base < 0)): return None
    else: val = sign(_base)

    try:
        result = val * math.pow(math.fabs(_base), _grade)
    except:
        result = None
    return result

def exp_(x):
    if (x is None): return None
    try:
        result = math.exp(x)
    except:
        result = None
    return result

def max_(x, y):
    if (x is None or y is None): return None
    if (x > y):
        return x
    else:
        return y

def min_(x, y):
    if (x is None or y is None): return None
    if (x > y):
        return y
    else:
        return x

def abs_(x):
    if (x is None): return None
    return math.fabs(x)

def asin_(x):
    if (x is None): return None
    try:
        result = math.asin(x)
    except:
        result = None
    return result

def sin_(x):
    if (x is None): return None
    try:
        result = math.sin(x)
    except:
        result = None
    return result

def sqrt_(x):
    if (x is None): return None
    try:
        result = math.sqrt(x)
    except:
        result = None
    return result

def log_(x):
    if (x is None): return None
    try:
        result = math.log(x)
    except:
        result = None
    return result

def log10_(x):
    if (x is None): return None
    try:
        result = math.log10(x)
    except:
        result = None
    return result
