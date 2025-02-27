import time

def summation(N):
    tot = 0
    for i in range(N):
        tot = tot+ i
    return(tot)

start_time = time.time()
summation(10000000)
end_time = time.time()

print('Execution time for summation:', end_time-start_time)