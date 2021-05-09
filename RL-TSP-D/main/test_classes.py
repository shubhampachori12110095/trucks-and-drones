

class SubClass:

    def __init__(self, value=1):
        self.value = value

    def add_one(self):
    	self.value +=1


class MetaClass:

    def __init__(self):
        self.sub_obj = SubClass()

meta_class = MetaClass()

print(meta_class.sub_obj.value)
meta_class.sub_obj.value = 3
print(meta_class.sub_obj.value)


class InheritClass(SubClass):

    def __init__(self, value):
        super().__init__(value)

inh_class = InheritClass(6)
print(inh_class.value)

list_1 = [SubClass(),SubClass(),SubClass()]

[thing.add_one() for thing in list_1]

print(list_1[0].value)