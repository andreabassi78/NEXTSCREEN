import time

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

