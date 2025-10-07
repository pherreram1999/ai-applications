import time
import tracemalloc


class Metrics:

    def __init__(self):
        self.time = 0
        self.memory_pico = 0
        self.current_memory = 0

    def __str__(self):
        return (f'tiempo: {self.time:.4f}s \n mem. pico: {self.memory_pico / 1024 / 1024:.2f} MB,' +
                f' mem. actual: {self.current_memory / 1024 / 1024:.2f} MB')

    pass

def trace(callback, *args) -> tuple[Metrics, tuple]:
    # ejecutamos y medimos lo datos de
    metrics = Metrics()
    tracemalloc.start()
    start = time.time()

    res = callback(*args)

    end = time.time()
    metrics.current_memory, metrics.memory_pico = tracemalloc.get_traced_memory()
    tracemalloc.stop()
    metrics.time = end - start
    return metrics, res
