import numpy as np
from numpy.random import random
from numpy import s_
from asp.config import PlatformDetector, ConfigReader
import asp.codegen.templating.template as AspTemplate
import asp.jit.asp_module as asp_module
from codepy.libraries import add_numpy, add_boost_python, add_cuda
from codepy.cgen import *
from codepy.cuda import CudaModule
import math
import sys
import time
from imp import find_module
from os.path import join

import pycasp

class SVMParameters(object):
    """
    The Python interface to the parameters of an SVM.
    """
    
    #TODO: which parameters have defaults?
    def __init__(self, kernelType, N,
            paramA = None, paramB = None, paramC = None):

        if paramA is not None:
            paramA = -paramA # ?

        if paramC is None:
            paramC = 3.0
        if paramA is None:
            paramA = 1.0/N
        if paramB is None:
            paramB = 0.0

        # determine kernel type and set the kernel parameters
        if kernelType == "linear":
            self.kernel_type = 0 
            self.gamma = 0.0 
            self.coef0 = 0.0 
            self.degree = 0.0 
        elif (kernelType == "gaussian"):
            self.kernel_type = 1 

            if paramA <= 0 or paramB < 0 :
                print "Invalid parameters"
                sys.exit()
            self.gamma = paramA
            self.coef0 = 0.0 
            self.degree = 0.0 

        elif kernelType == "polynomial":
            self.kernel_type = 2 
            if paramA <= 0 or paramB < 0 or paramC < 1.0:
                print "Invalid parameters"
                sys.exit()
            self.gamma = paramA
            self.coef0 = paramB
            self.degree = paramC # TODO: convert to int

        elif kernelType == "sigmoid":
            self.kernel_type = 3 
            if paramA <= 0 or paramB < 0 :
                print "Invalid parameters"
                sys.exit()
            self.gamma = paramA
            self.coef0 = paramB
            self.degree = 0.0 


        else:
            print "Unsupported kernel type. Please try one of the following: \
                  'linear', 'gaussian', 'polynomial', 'sigmoid'"
            sys.exit()

class SVM(object):
    """
    The specialized SVM abstraction. Based on the GMM specializer.
    """
    #Checking specializer configuration, compiler availability and platform features.
    #TODO: We track this stuff in singleton variables because this specializer only supports using one backend device for all SVM instances running from the same config file.
    platform = PlatformDetector()
    config = ConfigReader('SVM')
    cuda_device_id = config.get_option('cuda_device_id')
    template_path = config.get_option('template_path')
    autotune = config.get_option('autotune')
    names_of_backends_to_use = [config.get_option('name_of_backend_to_use')] #TODO: how to specify multiple backends in config file?
    use_cuda = False
    platform_info = {}
    if 'cuda' in names_of_backends_to_use:
        if 'nvcc' in platform.get_compilers() and platform.get_num_cuda_devices() > 0:
            use_cuda = True
            platform.set_cuda_device(cuda_device_id)
            platform_info['cuda'] = platform.get_cuda_info()
        else: print "WARNING: You asked for a CUDA backend but no compiler was found or no cuda device are detected by the CUDA driver."
            
    #Singleton ASP module shared by all instances of SVM. This tracks all the internal representation of specialized functions. 
    asp_mod = None

    def get_asp_mod(self): return SVM.asp_mod or self.initialize_asp_mod()

    #Internal defaults for the specializer. Application writers shouldn't have to know about these, but changing them might affect the API. 

    variant_param_default = { 'c++': {'dummy': ['1']},
        'cuda': {
            'num_blocks': ['128'],
            'num_threads': ['512'],
            'max_num_dimensions': ['10000']}
    }
    #TODO: incorporate this into the specializer...
    variant_param_autotune = { 'c++': {'dummy': ['1']},
        'cuda': {
            'num_blocks': ['16'],
            'num_threads': ['512'],
            'max_num_dimensions': ['10000']}
    }

    #Functions used to evaluate whether a particular code variant can be compiled or successfully run a particular input

    #TODO: incorporate this into the specializer...
    def cuda_compilable_limits(param_dict, gpu_info):
        #Determine if a code variant described by param_dict will compile on a device described by gpu_info
        tpb = int(gpu_info['max_threads_per_block'])
        shmem = int(gpu_info['max_shared_memory_per_block'])
        gpumem = int(gpu_info['total_mem'])
        blocks = int(param_dict['num_blocks'])
        threads = int(param_dict['num_threads'])
        max_d = int(param_dict['max_num_dimensions'])
        max_n = gpumem / (max_d*4)

        compilable = False

        if threads <= tpb: # and threads*4 < shmem: #TODO: KG, add shared memory constraints based on the SVM code
            compilable = True
        return compilable

    backend_compilable_limit_funcs = { 
        'c++':  lambda param_dict, device: True,
        'cuda': cuda_compilable_limits
    }

    def cuda_runable_limits(param_dict, gpu_info):
        #Return a lambda func that can determine whether the code variant described by param_dict can process the input args and kwargs
        tpb = int(gpu_info['max_threads_per_block'])
        shmem = int(gpu_info['max_shared_memory_per_block'])
        gpumem = int(gpu_info['total_mem'])
        threads = int(param_dict['num_threads'])
        max_d = int(param_dict['max_num_dimensions'])
        max_n = gpumem / (max_d*4)

        runnable = False 

        if threads <= tpb: # and threads*4 < shmem: #TODO: KG, add shared memory constraints based on the SVM code
            runnable = True
        return runnable

    backend_runable_limit_funcs = { 
        'c++':  lambda param_dict, device: lambda *args, **kwargs: True,
        'cuda': cuda_runable_limits
    }

    #Flags to keep track of memory allocations, singletons
    point_data_gpu_copy = None
    point_data_cpu_copy = None
    labels_gpu_copy = None
    labels_cpu_copy = None
    train_alphas_gpu_copy = None
    train_alphas_cpu_copy = None
    support_vectors_gpu_copy = None
    support_vectors_cpu_copy = None
    classify_alphas_gpu_copy = None
    classify_alphas_cpu_copy = None
    train_result_gpu_copy = None
    train_result_cpu_copy = None
    classify_result_gpu_copy = None
    classify_result_cpu_copy = None
    
    #Internal functions to allocate and deallocate component and event data on the CPU and GPU
    def internal_alloc_point_data(self, X):
        data_ptr = X.__array_interface__['data'][0]
        if SVM.point_data_cpu_copy != data_ptr:
            if SVM.point_data_cpu_copy is not None:
                self.internal_free_point_data()
            self.get_asp_mod().alloc_point_data_on_CPU(X)
            SVM.point_data_cpu_copy = X.__array_interface__['data'][0]
            if SVM.use_cuda:
                gpu_ptr = pycasp.get_GPU_pointer(data_ptr)
                SVM.point_data_gpu_copy = X.__array_interface__['data'][0]
                if gpu_ptr != 0: 
                    self.get_asp_mod().alloc_point_data_on_GPU_from_ptr(gpu_ptr, X.shape[0], X.shape[1])
                    SVM.point_data_gpu_copy = X.__array_interface__['data'][0]
                else:
                    self.get_asp_mod().alloc_point_data_on_GPU(X.shape[0], X.shape[1])
                    self.get_asp_mod().copy_point_data_CPU_to_GPU(X.shape[1])
                    SVM.point_data_gpu_copy = X.__array_interface__['data'][0]

    def internal_free_point_data(self):
        if SVM is None: return
        if SVM.point_data_cpu_copy is not None:
            SVM.point_data_cpu_copy = None      # deallocated by the GC
        if SVM.point_data_gpu_copy is not None:
            gpu_ptr = pycasp.get_GPU_pointer(SVM.point_data_gpu_copy)
            if gpu_ptr == 0: # only dealloc if this memory hasn't been allocated previously
                self.get_asp_mod().dealloc_point_data_on_GPU()
            SVM.point_data_gpu_copy = None
                
    def internal_alloc_labels(self, L):
        if SVM.point_data_cpu_copy != L.__array_interface__['data'][0]:
            if SVM.labels_cpu_copy is not None:
                self.internal_free_labels()
            self.get_asp_mod().alloc_labels_on_CPU(L)
            SVM.labels_cpu_copy = L.__array_interface__['data'][0]
            if SVM.use_cuda:
                self.get_asp_mod().alloc_labels_on_GPU(L.shape[0])
                self.get_asp_mod().copy_labels_CPU_to_GPU(L.shape[0])
                SVM.labels_gpu_copy = L.__array_interface__['data'][0]  
            
    def internal_free_labels(self):
        if SVM is None: return
        if SVM.labels_cpu_copy is not None:
            SVM.labels_cpu_copy = None      # dealloced by the GC
        if SVM.labels_gpu_copy is not None:
            self.get_asp_mod().dealloc_labels_on_GPU()
            SVM.labels_gpu_copy = None

    def internal_alloc_train_alphas(self, A):
        if SVM.point_data_cpu_copy != A.__array_interface__['data'][0]:
             if SVM.train_alphas_cpu_copy is not None:
                 self.internal_free_train_alphas()
             self.get_asp_mod().alloc_train_alphas_on_CPU(A)
             SVM.train_alphas_cpu_copy = A.__array_interface__['data'][0] 
             if SVM.use_cuda:
                 self.get_asp_mod().alloc_train_alphas_on_GPU(A.shape[0])
                 SVM.train_alphas_gpu_copy = A.__array_interface__['data'][0] 

    def internal_free_train_alphas(self):
        if SVM is None: return
        if SVM.train_alphas_cpu_copy is not None:
            SVM.train_alphas_cpu_copy = None     # dealloced by the GC
        if SVM.train_alphas_gpu_copy is not None:
            self.get_asp_mod().dealloc_train_alphas_on_GPU()
            SVM.train_alphas_gpu_copy = None
            
    def internal_alloc_classify_alphas(self, A):
        if SVM.point_data_cpu_copy != A.__array_interface__['data'][0]:
            if SVM.classify_alphas_cpu_copy is not None:
                self.internal_free_classify_alphas()
            self.get_asp_mod().alloc_classify_alphas_on_CPU(A)
            SVM.classify_alphas_cpu_copy = A.__array_interface__['data'][0] 
            if SVM.use_cuda:
                self.get_asp_mod().alloc_classify_alphas_on_GPU(A.shape[0])
                self.get_asp_mod().copy_classify_alphas_CPU_to_GPU(A.shape[0]);
                SVM.classify_alphas_gpu_copy = A.__array_interface__['data'][0] 

    def internal_free_classify_alphas(self):
        if SVM is None: return
        if SVM.classify_alphas_cpu_copy is not None:
            SVM.classify_alphas_cpu_copy = None     # dealloced by the GC
        if SVM.classify_alphas_gpu_copy is not None:
            self.get_asp_mod().dealloc_classify_alphas_on_GPU()
            SVM.classify_alphas_gpu_copy = None

    def internal_alloc_support_vectors(self, V):
        if SVM.point_data_cpu_copy != V.__array_interface__['data'][0]:
            if SVM.support_vectors_cpu_copy is not None:
                self.internal_free_support_vectors()
            self.get_asp_mod().alloc_support_vectors_on_CPU(V)
            SVM.support_vectors_cpu_copy = V.__array_interface__['data'][0] 
            if SVM.use_cuda:
                self.get_asp_mod().alloc_support_vectors_on_GPU(V.shape[0], V.shape[1])
                self.get_asp_mod().copy_support_vectors_CPU_to_GPU(V.shape[0], V.shape[1]);
                SVM.support_vectors_gpu_copy = V.__array_interface__['data'][0] 

    def internal_free_support_vectors(self):
        if SVM is None: return
        if SVM.support_vectors_cpu_copy is not None:
            SVM.support_vectors_cpu_copy = None     # dealloced by the GC
        if SVM.support_vectors_gpu_copy is not None:
            self.get_asp_mod().dealloc_support_vectors_on_GPU()
            SVM.support_vectors_gpu_copy = None

    def internal_alloc_train_result(self, R):
        if SVM.point_data_cpu_copy != R.__array_interface__['data'][0]:
            if SVM.train_result_cpu_copy is not None:
                self.internal_free_train_result()
            self.get_asp_mod().alloc_train_result_on_CPU(R)
            SVM.train_result_cpu_copy = R.__array_interface__['data'][0] 
            if SVM.use_cuda:
                self.get_asp_mod().alloc_train_result_on_GPU()
                SVM.train_result_gpu_copy = R.__array_interface__['data'][0] 

    def internal_free_train_result(self):
        if SVM is None: return
        if SVM.train_result_cpu_copy is not None:
            SVM.train_result_cpu_copy = None     # dealloced by the GC
        if SVM.train_result_gpu_copy is not None:
            self.get_asp_mod().dealloc_train_result_on_GPU()
            SVM.train_result_gpu_copy = None

    def internal_alloc_classify_result(self, R):
        if SVM.point_data_cpu_copy != R.__array_interface__['data'][0]:
            if SVM.classify_result_cpu_copy is not None:
                self.internal_free_classify_result()
            self.get_asp_mod().alloc_classify_result_on_CPU(R)
            SVM.classify_result_cpu_copy = R.__array_interface__['data'][0] 
            if SVM.use_cuda:
                self.get_asp_mod().alloc_classify_result_on_GPU(R.shape[0])
                SVM.classify_result_gpu_copy = R.__array_interface__['data'][0] 

    def internal_free_classify_result(self):
        if SVM is None: return
        if SVM.classify_result_cpu_copy is not None:
            SVM.classify_result_cpu_copy = None     # dealloced by the GC
        if SVM.classify_result_gpu_copy is not None:
            self.get_asp_mod().dealloc_classify_result_on_GPU()
            SVM.classify_result_gpu_copy = None

    def __init__(self): 
        self.names_of_backends_to_use = SVM.names_of_backends_to_use
        self.N = 0
        self.D = 0
        self.nSV = 0
        self.support_vectors = None
        self.alphas = None
        self.rho = None

    def __del__(self):
        self.internal_free_point_data()
        self.internal_free_labels()
        self.internal_free_train_alphas()
        self.internal_free_train_result()
        self.internal_free_support_vectors()
        self.internal_free_classify_alphas()
        self.internal_free_classify_result()

    #Called the first time a SVM instance tries to use a specialized function
    def initialize_asp_mod(self):
        # Create ASP module
        SVM.asp_mod = asp_module.ASPModule(use_cuda=SVM.use_cuda) 
        if SVM.use_cuda:
            self.insert_cache_controller_code_into_listed_modules(['c++', 'cuda'])
            self.insert_base_code_into_listed_modules(['c++'])
            self.insert_non_rendered_code_into_module()
            self.insert_rendered_code_into_cuda_module()
            SVM.asp_mod.backends['cuda'].toolchain.cflags.extend(["-Xcompiler","-fPIC", "-arch=sm_%s%s" % SVM.platform_info['cuda']['capability'] ])
            SVM.asp_mod.backends['c++'].toolchain.cflags.extend(["-lcublas"])
            SVM.asp_mod.backends['c++'].compilable = False # TODO: For now, must force ONLY cuda backend to compile

        #print SVM.asp_mod.generate()
        # Setup toolchain
        for name, mod in SVM.asp_mod.backends.iteritems():
            add_numpy(mod.toolchain)
            add_boost_python(mod.toolchain)
            if name in ['cuda']:
                add_cuda(mod.toolchain) 
        return SVM.asp_mod

    #Functions used in the template rendering process, specific to particular backends

    def insert_cuda_backend_render_func(self):
        #param_dict['supports_float32_atomic_add'] = SVM.platform_info['cuda']['supports_float32_atomic_add']
        cu_kern_tpl = AspTemplate.Template(filename = SVM.template_path + "templates/training/svm_cuda_kernels.mako")
        cu_kern_rend = cu_kern_tpl.render()
        SVM.asp_mod.add_to_module([Line(cu_kern_rend)],'cuda')
        c_decl_tpl = AspTemplate.Template(filename = SVM.template_path + "templates/training/svm_launch_decl.mako") 
        c_decl_rend  = c_decl_tpl.render()
        SVM.asp_mod.add_to_preamble(c_decl_rend,'c++') #TODO: <4.1 hack
        base_system_header_names = [ 'stdlib.h', 'stdio.h', 'sys/time.h']
        for header in base_system_header_names: 
            SVM.asp_mod.add_to_preamble([Include(header, True)], 'cuda')

    def insert_cache_controller_code_into_listed_modules(self, names_of_backends):

        cache_controller_t = '''
        #include <vector>
        #include <list>
        enum SelectionHeuristic {FIRSTORDER, SECONDORDER, RANDOM, ADAPTIVE};
            // From Cache.h
            
            class Cache {
              public:
                Cache(int nPointsIn, int cacheSizeIn);
                ~Cache();
                void findData(const int index, int &offset, bool &compute);
                void search(const int index, int &offset, bool &compute);
                void printCache();
            	void printStatistics();
            
              private:
                int nPoints;
                int cacheSize;
                class DirectoryEntry {
                  public:
                    enum {NEVER, EVICTED, INCACHE};
                    DirectoryEntry();
                    int status;
                    int location;
                    std::list<int>::iterator lruListEntry;
                };
            
                std::vector<DirectoryEntry> directory;
                std::list<int> lruList;
                int occupancy;
                int hits;
                int compulsoryMisses;
                int capacityMisses;
            };
            
            // From Controller.h
            
            using std::vector;
            
            class Controller {
             public:
              Controller(float initialGap, int currentMethodIn, int samplingIntervalIn, int problemSize);
              void addIteration(float gap);
              void print();
              int getMethod();
             private:
              bool adaptive;
              int samplingInterval;
              vector<float> progress;
              vector<int> method;
              int currentMethod;
              vector<float> rates;
              int timeSinceInspection;
              int inspectionPeriod;
              int beginningOfEpoch;
              int middleOfEpoch;
              int currentInspectionPhase;
              float filter(int begin, int end);
              float findRate(struct timeval* start, struct timeval* finish, int beginning, int end);
              struct timeval start;
              struct timeval mid;
              struct timeval finish;
            };
        '''
        for b_name in names_of_backends:
            SVM.asp_mod.add_to_preamble(cache_controller_t, b_name)

    def insert_base_code_into_listed_modules(self, names_of_backends):
        #Add code to all backends that is used by all backends
        c_base_tpl = AspTemplate.Template(filename = SVM.template_path + "templates/training/svm_base_helpers.mako")
        c_base_rend = c_base_tpl.render()

        base_system_header_names = [ 'stdlib.h', 'stdio.h', 'string.h', 'math.h', 'Python.h', 'sys/time.h', 'vector', 'list', 'numpy/arrayobject.h', 'cublas.h']
        for b_name in names_of_backends:
            for header in base_system_header_names: 
                SVM.asp_mod.add_to_preamble([Include(header, True)], b_name)
            SVM.asp_mod.add_to_preamble([Line(c_base_rend)],b_name)
            SVM.asp_mod.add_to_init("import_array();", b_name)

    def insert_non_rendered_code_into_module(self):
        #TODO: Move this back into insert_base_code_into_listed_modules for cuda 4.1
        names_of_helper_funcs = ["alloc_point_data_on_CPU",\
                                 "alloc_labels_on_CPU",\
                                 "alloc_train_alphas_on_CPU",\
                                 "alloc_classify_alphas_on_CPU",\
                                 "alloc_train_result_on_CPU",\
                                 "alloc_classify_result_on_CPU",
                                 "alloc_support_vectors_on_CPU",]
        for fname in names_of_helper_funcs:
            SVM.asp_mod.add_helper_function(fname, "", 'cuda')
        
        #Add bodies of helper functions
        c_base_tpl = AspTemplate.Template(filename = SVM.template_path + "templates/training/svm_cuda_host_helpers.mako")
        c_base_rend  = c_base_tpl.render()
        SVM.asp_mod.add_to_module([Line(c_base_rend)],'c++')
        cu_base_tpl = AspTemplate.Template(filename = SVM.template_path + "templates/training/svm_cuda_device_helpers.mako")
        cu_base_rend = cu_base_tpl.render()
        SVM.asp_mod.add_to_module([Line(cu_base_rend)],'cuda')
        #Add Boost interface links for helper functions
        names_of_cuda_helper_funcs = ["alloc_point_data_on_GPU",\
                                      "alloc_point_data_on_GPU_from_ptr",\
                                      "alloc_labels_on_GPU",\
                                      "alloc_train_alphas_on_GPU",\
                                      "alloc_train_result_on_GPU",\
                                      "alloc_classify_result_on_GPU",\
                                      "alloc_classify_alphas_on_GPU",\
                                      "alloc_support_vectors_on_GPU",\
                                      "copy_support_vectors_CPU_to_GPU",
                                      "copy_point_data_CPU_to_GPU",\
                                      "copy_labels_CPU_to_GPU",\
                                      "copy_classify_alphas_CPU_to_GPU",\
                                      "dealloc_point_data_on_GPU",\
                                      "dealloc_labels_on_GPU",\
                                      "dealloc_train_result_on_GPU",\
                                      "dealloc_classify_result_on_GPU",\
                                      "dealloc_train_alphas_on_GPU",\
                                      "dealloc_classify_alphas_on_GPU",\
                                      "dealloc_support_vectors_on_GPU",\
                                      "train", "classify"] 
        for fname in names_of_cuda_helper_funcs:
            SVM.asp_mod.add_helper_function(fname,"",'cuda')
        c_base_tpl = AspTemplate.Template(filename = SVM.template_path + "templates/training/cache_controller.mako")
        c_base_rend  = c_base_tpl.render()
        SVM.asp_mod.add_to_module([Line(c_base_rend)],'cuda')

    def insert_rendered_code_into_cuda_module(self):
        self.insert_cuda_backend_render_func()
        cu_base_tpl = AspTemplate.Template(filename = SVM.template_path + "templates/training/svm_train.mako")
        cu_base_rend = cu_base_tpl.render(num_blocks = 128, num_threads = 512)
        SVM.asp_mod.add_to_module([Line(cu_base_rend)],'c++')
        cu_base_tpl = AspTemplate.Template(filename = SVM.template_path + "templates/classification/svm_classify.mako")
        cu_base_rend = cu_base_tpl.render(num_blocks = 128, num_threads = 512)
        SVM.asp_mod.add_to_module([Line(cu_base_rend)],'c++')

    def train(self, input_data, labels, kernel_type, 
              paramA = None, paramB = None, paramC = None,
              heuristicMethod = None, tolerance = None, cost = None, epsilon = None):
        """
        Get training data.
        """
        self.N = input_data.shape[0] 
        self.D = input_data.shape[1] 

        """
        Allocate helper structures
        """
        training_alpha = np.empty(self.N, dtype=np.float32)
        result = np.empty(8, dtype=np.float32)
        
        """
        Setup SVM parameters.
        """
        self.kernel_params = SVMParameters(kernel_type, self.N, paramA, paramB, paramC)
        self.heuristic = heuristicMethod if heuristicMethod is not None else 3 #Adaptive 
        self.cost = cost if cost is not None else 10.0 
        self.tolerance = tolerance if tolerance is not None else 1e-3
        self.epsilon = epsilon if epsilon is not None else 1e-5 

        """
        Allocate data structures.
        """
        self.internal_alloc_point_data(input_data)
        self.internal_alloc_labels(labels)
        self.internal_alloc_train_alphas(training_alpha)
        self.internal_alloc_train_result(result)

        """
        Train the SVM on the data.
        """
        train_result = self.get_asp_mod().train(self.N, self.D,
                                 self.kernel_params.kernel_type,
                                 self.kernel_params.gamma,
                                 self.kernel_params.coef0,
                                 self.kernel_params.degree,
                                 self.cost,
                                 self.heuristic, self.epsilon,
                                 self.tolerance)
        
        self.support_vectors, self.alphas, self.rho, self.nSV = train_result
        self.support_vectors = self.support_vectors.reshape(self.nSV, self.D)

        return 

    def classify(self, input_data, labels):
        """
        Get testing data.
        """
        self.N = input_data.shape[0] 
        self.D = input_data.shape[1] 
        
        """
        Allocate result array.
        """
        classify_result = np.empty(self.N, dtype=np.float32)

        """
        Check if all data structures have been allocated during training.
        """
        if (self.support_vectors is None) or (self.alphas is None) or (self.rho is None) or (self.nSV == 0):
            print "Error in calling classify: svm parameters not present."
            print "Please call train() before calling classify()"
            sys.exit()
        
        """
        Allocate data structures.
        """
        self.internal_alloc_point_data(input_data)
        self.internal_alloc_labels(labels)
        self.internal_alloc_support_vectors(self.support_vectors)
        self.internal_alloc_classify_alphas(self.alphas)
        self.internal_alloc_classify_result(classify_result)

        """
        Classify testing data.
        """

        # writes to classify_result
        self.get_asp_mod().classify(self.N, self.D,
                                    self.nSV,
                                    self.kernel_params.kernel_type,
                                    self.kernel_params.gamma,
                                    self.kernel_params.coef0,
                                    self.kernel_params.degree,
                                    self.rho)

        return classify_result
