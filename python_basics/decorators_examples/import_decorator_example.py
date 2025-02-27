from timer_decorator import add_timer

@add_timer
def summation(N):
    tot = 0
    for i in range(N):
        tot = tot + i
    return(tot)

summation(10000000)