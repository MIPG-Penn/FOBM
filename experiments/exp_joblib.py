#%%
import time

def slow_power(x, p):
    time.sleep(1)
    return x ** p

#%%
%time [slow_power(i, 5) for i in range(10)]

#%%
import joblib
from joblib import Parallel, delayed

number_of_cpu = joblib.cpu_count()

delayed_funcs = [delayed(slow_power)(i, 5) for i in range(10)]
parallel_pool = Parallel(n_jobs=number_of_cpu)

%time parallel_pool(delayed_funcs)

#%%
from tqdm.contrib.concurrent import process_map  # or thread_map
import time

def _foo(my_number):
   square = my_number * my_number
   time.sleep(1)
   return square 

if __name__ == '__main__':
   r = process_map(_foo, range(0, 30), max_workers=2)

#%%
from tqdm import tqdm
from multiprocessing import Pool
from functools import partial
import time
import random

def func_single_argument(n):
    time.sleep(0.5)
    return n

def func_multiple_argument(n, m, *args, **kwargs):
    time.sleep(0.5)
    return n, m

def run_imap_multiprocessing(func, argument_list, num_processes):

    pool = Pool(processes=num_processes)

    result_list_tqdm = []
    for result in tqdm(pool.imap(func=func, iterable=argument_list), total=len(argument_list)):
        result_list_tqdm.append(result)

    return result_list_tqdm

num_processes = 10
num_jobs = 100
random_seed = 0
random.seed(random_seed) 

# imap, imap_unordered
# It only support functions with one dynamic argument
func = func_single_argument
argument_list = [random.randint(0, 100) for _ in range(num_jobs)]
print("Running imap multiprocessing for single-argument functions ...")
result_list = run_imap_multiprocessing(func=func, argument_list=argument_list, num_processes=num_processes)