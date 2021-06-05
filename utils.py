# from types import FunctionType
# from functools import wraps

# def wrapper(method):
#     @wraps(method)
#     def wrapped(*args, **kwrds):
#     #   ... <do something to/with "method" or the result of calling it>
#     return wrapped

# def CopiedType(wrapper):
from inspect import signature
from abc import ABCMeta

import numpy as np

def joint_index(varlist):
    """Given sets/variables/lists [Unit, A, B, C], reutrns
    np.array([ (⋆, a0, b0, c0), .... (⋆, a_m, b_n, c_l) ]) 
    """
    return np.rollaxis(np.stack(np.meshgrid(*[np.array(*X) for X in varlist],  indexing='ij')), 0, len(varlist)+1)
    
def dictwo(d1, delkeys):
    return {k:v for k,v in d1.items() if k not in delkeys }
    
""" https://stackoverflow.com/a/47431859
only flattens tuples,"""
def flatten_tuples(l, start_depth=-1, end_depth=float('inf')):
    for el in l:
        if isinstance(el, tuple):
            rec = flatten_tuples(el, start_depth-1, end_depth-1)
            if start_depth <= 0 and end_depth > 0:
                for sub in rec:
                    yield sub
            else:
                yield tuple(rec)
        else:
            yield el

# if False:
#     t = ('a','b','c')
#     list(flatten_tuples((t,(t,t)),0,1))
# 
#     list(flatten_tuples((1,(2,(3,(4,(5,6),4)))),-1))
#     list(flatten_tuples((1,(2,(3,(4,(5,6),4)))),0,0))
#     list(flatten_tuples((1,(2,(3,(4,(5,6),4)))),0,1))
#     list(flatten_tuples((1,(2,(3,(4,(5,6),4)))),0,2))
#     list(flatten_tuples((1,(2,(3,(4,(5,6),4)))),0,3))
#     list(flatten_tuples((1,(2,(3,(4,(5,6),4)))),0,4))
#     list(flatten_tuples((1,(2,(3,(4,(5,6),4)))),1,2))
#     list(flatten_tuples((1,(2,(3,(4,(5,6),4)))),2,3))
#     list(flatten_tuples((1,(2,(3,(4,(5,6),4)))),3,4))
            
def tuple_depth(l):
    if not isinstance(l, tuple):
        return 0
        
    return 1+ max(( tuple_depth(el) for el in l if isinstance(el, tuple)), default=0)

class CopiedType(type):    
    def __new__(meta, classname, bases, classDict):
        newClassDict = dict(classDict)
        # for attributeName, attribute in classDict.items():
        #     if isinstance(attribute, FunctionType):
        #         # replace it with a wrapped version
        #         attribute = wrapper(attribute)
        #     newClassDict[attributeName] = attribute
        
        found_dict = {}
        
        for b in bases:
            for funcname, func in b.__dict__.items():
                if funcname in classDict:
                    continue
                if found_dict.get(funcname,False):
                    del newClassDict[funcname]
                else:
                    newClassDict[funcname] = func
                                    
                found_dict[funcname] = True
                
    
        if 'PARAMS' in classDict:
            class_params = classDict['PARAMS']
            custom_init = None
            params = set(class_params)
            # firstArgs = {}
            # offset = 0
            
            if '__init__' in classDict:
                custom_init = classDict['__init__']
                old_init_params = signature(custom_init).parameters
                # offset = max((p for p in params if type(p) == int), default=-1)+1
                
                params |=  set(pn for pn,p in old_init_params.items() \
                    if p.kind == p.POSITIONAL_OR_KEYWORD or p.kind == p.KEYWORD_ONLY)
                
                # firstArgs = {pn:p for pn,p in old_init_params.items() \
                #     if (p.kind == p.POSITIONAL_OR_KEYWORD or p.kind == p.POSITIONAL_ONLY) \
                #     and p.default is p.empty }
                # 
                # print(firstArgs)
                
            for b in bases:
                if '__init__' in b.__dict__:
                    base_init = b;
                    break
            
            def alterinit(self, *args, **kwargs):
                # kwargs_filtered = {k : v for k,v in kwargs.items() if k not in params }
                
                # print("super: ", super().__init__)
                # print("super globals: ", super(globals()[classname], self).__init__)
                # print("base: ", bases[0].__init__)
                metap = {"debug"}
                
                if 'debug' in kwargs and 'print' in kwargs['debug']:
                    print('ARGS: ', args, '\nKWARGS:', kwargs)
                
                base_init.__init__(self, *args[:], **dictwo(kwargs, params | metap ))  
                if custom_init:
                    custom_init(self, *args, **dictwo(kwargs,class_params | metap))

                for name in params:
                    if name in kwargs:
                        setattr(self, name, kwargs[name])

            alterinit.__name__ = '__init__'
            alterinit.__doc__ = '* BASE INIT: ' + base_init.__doc__ 
            if custom_init and custom_init.__doc__:
                alterinit.__doc__ += "\n\nCUSTOM INIT: " + custom_init.__doc__

            newClassDict['__init__'] = alterinit
        
        return type.__new__(meta, classname, bases, newClassDict)


class CopiedABC(CopiedType, ABCMeta): pass
### TESTS for CopiedType
# class A(set, metaclass=CopiedType):
#     # def __init__(self): 
#     #     print("A init")
#     PARAMS = {"z", "x"}
# 
#     def a(self, aval):
#         print(aval);
# 
#     def __mul__(self, other):
#         return set((a,b) for b in other for a in self)
# 
# 
# A.__init__
# 
# a = A( {0,1}, z = 3)
# b = A({0,2})
