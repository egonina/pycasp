import asp.jit.asp_module as asp_module
import numpy as np
from em import *
import pickle
import sys

param_type_map = {
        'num_blocks_estep': ('cardinal','variant'),
        'num_threads_estep': ('cardinal','variant'),
        'num_threads_mstep': ('cardinal','variant'),
        'num_event_blocks': ('cardinal','variant'),
        'max_num_dimensions': ('cardinal','variant'),
        'max_num_components': ('cardinal','variant'),
        'max_num_dimensions_covar_v3': ('cardinal','variant'),
        'max_num_components_covar_v3': ('cardinal','variant'),
        'diag_only': ('binary','variant'),
        'max_iters': ('cardinal','variant'),
        'min_iters': ('cardinal','variant'),
        'covar_version_name': ('nominal','variant'),
        'supports_32b_floating_point_atomics': ('nominal','machine'),
        'max_xy_grid_dim': ('cardinal','machine'),
        'max_threads_per_block': ('cardinal','machine'),
        'max_shared_memory_capacity_per_SM': ('cardinal','machine')
}

if __name__ == '__main__':  
        ifile_name = sys.argv[1]
        ofile_name = sys.argv[2]
        func_name = sys.argv[3]
        device_id = sys.argv[4]
        gmm = GMM(1,1)
        mod = gmm.get_asp_mod()
        mod.restore_method_timings(func_name,ifile_name)
        var_names = mod.compiled_methods[func_name].v_id_list
        param_names = mod.compiled_methods[func_name].param_names
        var_times = mod.compiled_methods[func_name].database.variant_times
        f = file(ofile_name, 'a')
        f.write("Heading, Function Name, Device Name, Input Params,,,Variant Params"+","*len(param_names)+"Time\n")
        f.write("Name,function,device,M,D,N,%s,Time\n" % ','.join(param_names))
        f.write("Type,nominal,nominal,cardinal,cardinal,cardinal,%s,real\n" % 
                ','.join([param_type_map.get(n,'unknown')[0] for n in param_names]))
        f.write("Prefix,problem,machine,problem,problem,problem,%s,performance\n" % 
                ','.join([param_type_map.get(n,'unknown')[1] for n in param_names]))
        for size, times in var_times.items():
            for name in var_names:
                time = times[name]
                f.write(",%s,%s,%s,%s,%s\n" % ( func_name, 
                                                device_id,  
                                                ','.join([str(p) for p in size[1:]]),
                                                ','.join(name.split('_')[1:]),
                                                time ) )
        f.close()

