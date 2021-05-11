'''

[thing.add() for thing in list_sub]

entspricht:

for thing in list_sub:
    thing.add()

'''

class Tracer:
    def __init__(self, value):
        self.value = value

tracer = Tracer(0)
#-------------------------------------------------------------------


class BaseClass:

    def __init__(self, obj):
        self.obj = obj

    def add(self):
        self.obj.value +=1

list_base = [BaseClass(tracer),BaseClass(tracer),BaseClass(tracer)]

[thing.add() for thing in list_base]
print(tracer.value)
#-------------------------------------------------------------------


class MetaClass:

    def __init__(self,tracer):
        self.sub_obj = BaseClass(tracer)

list_meta = [MetaClass(tracer),MetaClass(tracer),MetaClass(tracer)]

[thing.sub_obj.add() for thing in list_meta]
print(tracer.value)
#-------------------------------------------------------------------


class InheritClass(BaseClass):

    def __init__(self, obj):
        super().__init__(obj)

    def add(self):
        self.obj.value +=2

    def add_value(self,value):
        self.obj.value +=value

list_inherit = [InheritClass(tracer),InheritClass(tracer),InheritClass(tracer)]

[thing.add() for thing in list_inherit]
print(tracer.value)

[thing.add_value(5) for thing in list_inherit]
print(tracer.value)
#-------------------------------------------------------------------
