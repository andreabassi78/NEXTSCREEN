import time

def add_timer(function):
    """Basic function decorator"""

    def inner(*args,**kwargs):
        start_time = time.time()
        result = function(*args,**kwargs)
        end_time = time.time()
        print ('Execution time:', end_time-start_time)
        return result

    return inner    

@add_timer
def summation(N):
    tot = 0
    for i in range(N):
        tot = tot+ i
    return(tot)

res = summation(10000000)
#print(res)