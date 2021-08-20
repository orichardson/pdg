import numpy as np
import pandas as pd
from dist import RawJointDist as RJD
from environs import Env
from collections import defaultdict
# from collections import frozenset as fz

from itertools import chain
from inspect import getsource

def _has2(v):
    return isinstance(v, tuple) and len(v) == 2
def _mixed2dict( mixed_selector, default ):
    return dict((x if _has2(x) else (x, default)) for x in mixed_selector)
    

def valid_selector(s):
    if _has2(s):
        return valid_selector(s[0])
    return not (isinstance(s,str) and s[0] == '_')

def fz(*tags, **attrs):
    return frozenset(chain(tags,
                    filter(valid_selector, attrs.items())))
                    
                    
def prettify_selector( sel ):
    return '; '.join(
        (str(s[0])+"="+str(s[1]) if isinstance(s,tuple) and len(s)==2 else str(s)) for s in sel)

def orthogonalize_selectors( selectors ): 
    allkeys = set()
    for sel in selectors:
        d = _mixed2dict(sel, True)
        allkeys.update(d)
        # { a : 3, b: 4, tag : True }   
        
    # data = 
    
    raise NotImplementedError()
    
# Might make it easier to manipulate selectors. 
class Selector:
    def __init__(self, *tags, **attrs):
        self._sel = fz(*tags, **attrs)
        
    def __repr__(self):
        return prettify_selector(self._sel)
        
    def todict(self):
        return _mixed2dict(self._sel)
        

class TensorLibrary:
    def __init__(self, shape=(-1,), decoder=None):
        # self.tensordata = dists
        self.ushape = shape
        self.decoder = decoder
        self.tensordata = {} # frozenset( str | (k:v) )  =>  ℝ(ushape)    
        # self.ushape = M.genΔ(repr=store_repr).data.reshape(shape).shape

    def __getattr__(self, name):
        return getattr(LView(self), name)
        # if name[0] == '_':
            # pass
        # return View(self).__getattr__(name)
        
    # def __iadd__(self, other):
        # pass
        
    def items(self):
        return self.tensordata.items()
        
    def keys(self):
        return self.tensordata.keys()
        

    def __call__(self, *posspec, **kwspec):
        return LView(self)(*posspec, **kwspec)


    def _validate(self, value):
        if self.shape:
            try:
                argvalue = value
                if isinstance(value, RJD): value = value.data
                elif isinstance(value, Env): value = value.TT
                
                value = np.asarray(value) #try this
                return value.reshape(self.ushape)
            except:
                raise TypeError("Only tensors of shape "+str(self.ushape)+" allowed; \"", argvalue, "\" could not be reshaped this way")
        else:
            return value

    def _decode(self, stored_tensor):
        if self.decoder:
            return self.decoder(stored_tensor)
        else:
            return stored_tensor

    def __setitem__(self, key, val):
        self.tensordata[fz(key)] = self._validate(val)
        
    def copy(self):
        tl = TensorLibrary(self.shape, self.decoder)
        tl.tensordata = dict(self.tensordata)
        return tl
        
    # def j
    #TODO: niciefy this
    # def __repr__(self):
    #     return "<DistLib with keys {%s}>"%s
    def __iter__(self):
        return iter(LView(self))

    def __pos__(self):
        return +LView(self)
        
    def __len__(self):
        return len(self.tensordata)
        
    # def __repr__(self):
    #     return 



class LView:
    def __init__(self, library, *selector, **kwselect):
        self._lib = library
        self._most_recent_tag = selector[-1] if len(selector) > 0 else None
        self._filters = kwselect.get('_filters', [])
        self._sel  = fz(*selector, **kwselect)
        self._cached = list(self._consist_from_lib())


    def _consist_from_lib(self):
        for k,d in self._lib.tensordata.items():
            # if self._sel.issubset(k) \
            if all(((t in k or t[0] in k) if _has2(t) else (t in k)) for t in self._sel ) \
                    and all(f(k) for f in self._filters):
                yield k,d

    def along(self, axis, return_tags=False):
        """
        Allows you to simultaneously filter by an attribute and sort by it, 
        optionally in decending order.
        
        E.g., `store.along('-x')`
        # returns an iterator of tensors with the x attribute, sorted in reverse
        """
        reverse=False
        if isinstance(axis,str) and axis[0] in '+-':
            reverse = (axis[0] == '-')
            axis = axis[1:]

        values = []
        for S,d in self.raw:
             v = next((atom[1] for atom in S if atom[0] == axis), None)
             if v is not None:
                 values.append( (v,(S,self._lib._decode(d))) )

        return (((S,d) if return_tags else d) for v,(S,d) in sorted(values, reverse=reverse))

    def values_for_key(self, key):
        found = set()
        for K in self.matches:
            for k in K:
                if _has2(k) and k[0] == key:
                    if k[1] not in found:
                        found.add(k[1])
                        yield k[1]
                    continue 
        

    def filter(self, f):
        return LView(self._lib, *self._sel, _filters=[*self._filters, f])
        
    @property
    def tags(self):
        return set( k for S in self.matches for k in _mixed2dict(S, None).keys() )

    @property
    def tensors(self):
        for s,d in self:
            yield d
            
    @property 
    def df(self):
        mykeys = _mixed2dict(self._sel, None).keys()
        allkeys = set()
        for sel in self.matches:
            d = _mixed2dict(sel, True)
            allkeys.update(d.keys() - mykeys)
            
        # Now, iterate again, putting in the data
        names = sorted(allkeys)
        # dims = [] if self._lib.ushape is None else ["x_%d"%i for i,d in enumerate(self._lib.ushape) if d > 1]
        
        idx_tuples = []
        framedata = []
        
        # print('NAMES', names)
        
        for sel, t in self.raw:
            d = _mixed2dict(sel, True)
            idx = tuple(d.get(n, None) for n in names)
            # print(idx)
            idx_tuples.append(idx)
            framedata.append(dict({n : d.get(n,None) for n in names}, tensor=t))
            
        # INDEX = pd.MultiIndex.from_tuples(idx_tuples, names=names)
        # print(INDEX)
        
        # return INDEX
        return pd.DataFrame(framedata)
            
            
            
    def dataframe_by_attrs(self, axis1, axis2, agglomerator=np.mean, prettify=True):
        """
        For instance,
            (a=2, b=3, c, d=3)   [3,2,1]
            (a=1, b=7, c', d=1)  [1,2,3]
            (a=1, b=3, d=4)      [1,1,1]
            (a=1, b=3, d=7, e)   [3,3,3]
        
        .dataframe_by_attrs("a", "b", np.mean)
        
        gives
               __b = 3___b = 7 ___
        a = 1 |   
        a = 2 |  2       nan
        """
        dictofdicts = defaultdict(lambda: defaultdict(list))
   
        for S, t in self.raw:
            sdict = _mixed2dict(S, None)
            try:
                dictofdicts[sdict[axis1]][sdict[axis2]].append( t )
            except KeyError:
                pass
            
        for k1 in dictofdicts:
            for k2 in dictofdicts[k1]:
                dictofdicts[k1][k2] = agglomerator(dictofdicts[k1][k2])

        df= pd.DataFrame(dictofdicts)
        
        if prettify:
            for labels,axis in zip([df.index, df.columns], [0,1]):
                try:
                    common = frozenset.intersection(*labels)
                    df.rename({ L : prettify_selector(L-common) for L in labels}, axis=axis, inplace=True)
                except TypeError:
                    pass # No frozenset to intersect; prettify procedure does not apply.
        return df

    @property
    def matches(self):
        for s,d in self.raw:
            yield s

    def without(self, tag, **kwargs):
        return self.filter(lambda taglist: tag not in _mixed2dict(taglist, ...))

    def set(self, dist):
        self._lib.tensordata[self._sel] = self._lib._validate(dist)
        
    def tagAll(self, *tags, **kwtags):
        newdict = {}
        for S,t in self.raw:
            # make sure new values are overriden
            S_preempt_duplicates = [ k for k in S if not(_has2(k) and valid_selector(k) and  k[0] in kwtags) ]
            
            newS = frozenset(chain(S_preempt_duplicates,tags,\
                filter(valid_selector, kwtags.items())))
            # newS = S.union(tags, filter(valid_selector, kwtags.items()))
            newdict[newS] = self._lib.tensordata[S] 
            del self._lib.tensordata[S]
        
        # print(newdict)
        self._lib.tensordata.update(newdict)
        self._sel = self._sel.union(tags, filter(valid_selector, kwtags.items()))

    # def __iadd__(self, datum):
    #     self.set(datum)

    @property
    def raw(self):
        for s,d in (self._cached if self._cached else self._consist_from_lib()):
            yield s,d
    
    def __iter__(self):
        for s,d in (self._cached if self._cached else self._consist_from_lib()):
            yield s, self._lib._decode(d)

    def __pos__(self):
        lubs = []
        for s,d in self.raw:
            if all(s.issubset(l) for l in lubs):
                lubs = [ s ]
            elif not any(l.issubset(s) for l in lubs):
                lubs.append(s)
        if len(lubs) != 1:
            raise ValueError("No minimal distribution in this view! (there are %d)"%len(lubs))

        return self._lib._decode(self._lib.tensordata[lubs[0]])
        # return self._lib.tensordata[self._sel]
        # return next(iter(self.μs))

    def __repr__(self):
        selectorstr = prettify_selector(self._sel)
        if self._filters:
            selectorstr += " | <%d filters>" % len(self._filters)

        return "LView { %s } (%d matches)" % (selectorstr, len(self._cached))


    def __getattr__(self, name):
        if frozenset([*self._sel, name]) in self._lib.tensordata:
            return self._lib._decode(self._lib.tensordata[name])

        if name[0] == '_':
            raise AttributeError

        nextview = LView(self._lib, *self._sel, name)
        # if len(nextview._cached) == 0:
        #     raise AttributeError("No distributions matching spec `%s` in library"%str(name))
        return nextview

    def __call__(self, *tags, **kwspec):
        filters = list(self._filters)
        if len(tags) == 1 and hasattr(tags[0], '__call__') and len(self._sel) > 0:
            def interpreted_filter(taglist):
                val = dict(filter(_has2, taglist)).get(self._most_recent_tag, None)
                return tags[0](val) if val is not None else (self._most_recent_tag in taglist)
                # TODO: LOOK UP [self._sel[-1]]
            interpreted_filter.__doc__ = getsource(tags[0])
            filters.append(interpreted_filter)
            T = self._sel - {self._most_recent_tag}
        else:
            T = [*self._sel, *tags]

        nextview = LView(self._lib,  *T, _filters=filters, **kwspec)
        # if len(nextview._cached) == 0:
        #     raise ValueError("No distributions matching spec `%s` in library"%str(name))
        return nextview

    # Before uncommenting: either make underscores special, change
    # the constructor where things are initialized, or enable a flag after
    # construction.
    # def __setattr__(self, key, dist):
    #     self._lib[name, frozenset(*self._sel, key)] = dist

    # This doesn't work.
    # def __set__(self, obj, value):
    #     print("__set__ called with: ", self, obj, value)
    #     self._lib[self._sel] = value

"""
def clean_selector():
    shared1 = None
    shared2 = None

    for S, t in self.raw:
        sdict = _mixed2dict(S, None)

        if shared1 is None: 
            shared1 = sdict[axis1]
        else:
            shared1 &= sdict[axis1]

        if shared2 is None: 
            shared2 = sdict[axis2]
        else:
            shared2 &= sdict[axis2]
"""
