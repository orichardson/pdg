import itertools
import abc

from . import utils

class RV(abc.ABC):
    pass
    # @abc.abstractproperty
    # def vals(self):

class ConditionRequest(object, metaclass=utils.CopiedType):
    PARAMS = {"target", "given"}

    @property
    def name(self):
        return self.target.name + " | "+ self.given.name


class Variable(set, metaclass=utils.CopiedType):
    PARAMS = {'name', 'default_value'}

    def __init__(self, vals):
        # super init inserted by metaclass
        self._ordered_set = list(vals)
        self.structure = []
        self |= set(vals)

    @property
    def is1(self):
        return self == Unit

    @staticmethod
    def product( *varis):
        if len(varis) == 1:
            if isinstance(varis[0], Variable):
                return varis[0]
            return Variable.product(*varis[0])
        elif len(varis) == 0:
            return Unit
            
        kwargs = {"default_value" : (), "name" : () }

        for v in varis:
            for key in list(kwargs.keys()):
                if hasattr(v, key):
                    kwargs[key] = (*kwargs[key], getattr(v,key))
                else:
                    del kwargs[key]

        if 'name' in kwargs:
            kwargs['name'] = "×".join(kwargs['name']) if len(varis) else '1'

        joint = Variable(list(itertools.product(*(tuple(v.ordered) for v in varis))) , **kwargs)
        # previously: wanted to keep all structure. Now, keep it heirarchically. If structure gets
        #   changed
        # joint.structure = [st for V in varis for st in V.structure] + [JointStructure(joint, *varis)]
        joint.structure = [ JointStructure(joint, *varis) ]
        # joint.structure = [*self.structure, *other.structure, JointStructure(joint, self, other)]
        # joint =  Variable([(a,b) for a in self.ordered for b in other.ordered ], **kwargs)

        return joint

    @property
    def all_substructures(self):
        for s in self.structure:
            if isinstance(s,JointStructure):
                for v in s.components:
                    yield from v.all_substructures

    def __and__(self, other):
        return Variable.product(self,other)
        # kwargs = {}
        # if hasattr(self, 'default_value') and hasattr(other, 'default_value'):
        #     kwargs['default_value'] = (self.default_value, other.default_value)
        # if hasattr(self, 'name') and hasattr(other, 'name'):
        #     kwargs['name'] = (self.name + "×" + other.name)
        #
        # joint =  Variable([(a,b) for a in self.ordered for b in other.ordered ], **kwargs)
        # joint.structure = [*self.structure, *other.structure, JointStructure(joint, self, other)]
        # return joint

    # def __pow__(self, num):
    #     joint = Variable

    def __ior__(self, other):
        newelts = [ o for o in other if not o in self._ordered_set ]
        self._ordered_set = self._ordered_set + newelts
        self.update(other)
        return self

    def __repr__(self):
        return "Var %s {%s}" % ( self.name if hasattr(self, 'name') else '', ', '.join(repr(v) for v in self.ordered) )

    def copy(self) -> 'Variable':
        duplicate = Variable(self.ordered, **{k:v for k,v in self.__dict__.items() if k in Variable.PARAMS})
        duplicate.structure = [*self.structure]
        return duplicate
    # with a variable V taking v, can write
    # V.v


    """ conditioning """
    def __or__(self, other):
        return ConditionRequest(target=self,given=other)

    def __eq__(self, other):
        if isinstance(other,Variable):
            named = hasattr(self,"name")
            nameeq = named == hasattr(other,"name")
            return set.__eq__(self,other) and nameeq and (self.name == other.name if named else True)

        # return isinstance(other, Variable) and set.__eq__(self, other) and (
        #         self.name == other.name if hasattr(self,"name") else True)

    def __hash__(self):
        return hash( (frozenset(self), ) + ((self.name,) if hasattr(self,'name') else ()))


    def split(self, atomic=True):
        for s in self.structure:
            if isinstance(s, JointStructure):
                for V in s.components:
                    if not (atomic and '×' in V.name):
                        yield V

    @property
    def atoms(self):
        if self == Unit:
            return
            
        js = [s for s in self.structure if isinstance(s,JointStructure)]
        if len(js) == 0:
            yield self
        else:
            for s in js:
                for v in s.components:
                    yield from v.atoms

    @property
    def ordered(self):
        self._ordered_set = [x for x in self._ordered_set if x in self] + \
            [y for y in self if y not in self._ordered_set]
        return self._ordered_set

    # @property
    # def pd_index(self):
    #     pass

    @classmethod
    def alph(cls, name : str, n : int):
        nl = name.lower()
        return cls([nl+str(i) for i in range(n)], default_value=nl+"0", name=name)
# V = Variable([3, 10, 2], name='V')
# (V*V).name
# α = β / γ

def binvar(name : str) -> Variable:
    nl = name.lower()
    return Variable([nl, "~"+nl], default_value=nl, name=name)

Unit = Variable('⋆', default_value='⋆', name='1')


class JointStructure:
    def __init__(self, all, *components ):
        self.joint = all
        self.components = components

    def __repr__(self):
        # return f"Joint [{ ' '.join(v.name for v in self.components) }]"
        return "Joint ["+ ' '.join(v.name for v in self.components) +"]"

    def gen_cpts_for(self, pdg):
        from .dist import CPT

        if self.joint.name in pdg.vars:
            for i,V in enumerate(self.components):
                if V.name in pdg.vars:
                    yield "π%d"%(i+1), CPT.det(self.joint, V, {v: v[i] for v in self.joint})

            # hasL = self.left.name in pdg.vars
            # hasR = self.right.name in pdg.vars
            #
            # if hasL:
            #     yield "π1", CPT.det(self.joint, self.left, {v: v[0] for v in self.joint})
            # if hasR:
            #     yield "π2", CPT.det(self.joint, self.right, {v: v[1] for v in self.joint})
            #
            # Maybe also: universal property
            # generate CPT going into joint for every pair
            # going into CPT, from any other variable.
            # TODO later
