class Person:

    def __init__(self,name,age):
        self.name = name
        self.age = age
        self.job = 'unemployed'

    def find_job(self, new_job):
        self.job = new_job

    def birthday(self):
        self.age = self.age + 1

Marco = Person('Marco',26)
Marco.find_job('Phd student')
Marco.birthday()

print(dir(Marco))


    

