import time

def _add_timer(function):
    """Basic function decorator to mesaure the execution time of a function
    works only if the function get 1 single parameter.
    """ 
    def inner(params):
        start_time = time.time() 
        result = function(params) 
        end_time = time.time() 
        print('Execution time, calculated with add_timer:', end_time-start_time) 
        return result
    
    return inner

def add_timer(function):
    """Function decorator to mesaure the execution time of a function or a method.
    """ 
    def inner(*args,**kwargs):
        
        print(f'\nStarting method "{function.__name__}" ...') 
        start_time = time.time() 
        result = function(*args, **kwargs) 
        end_time = time.time() 
        print(f'Execution time for method "{function.__name__}": {end_time-start_time:.6f} s') 
        
        return result
    inner.__name__ = function.__name__
    return inner

@add_timer
def factorial(N):
    tot = 0
    for i in range(N):
        tot = tot + i
    return(tot)

if __name__ == '__main__':
    t0 = time.time()
    #add_timer(factorial)(10000000)
    factorial(10000000)
    t1 = time.time()
    print('Execution time:', t1-t0)