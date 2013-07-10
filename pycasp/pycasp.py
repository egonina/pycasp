# code to keep track of memory allocations
data_map = {}

def add_to_data_map(key, value):
    if key not in data_map.keys():
        #print "PYCASP: memory add, ", key, ":", value
        data_map[key] = value

def get_GPU_pointer(key):
    val = 0 
    if key in data_map.keys():
        val = data_map[key]
    return val
