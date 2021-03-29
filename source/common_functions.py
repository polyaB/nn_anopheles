from hashlib import sha224

def str2hash(s):
    h = str(int(sha224(s.encode()).hexdigest(), 16) % (10 ** 10))
    return h