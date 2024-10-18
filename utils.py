from flex.pool import FlexPool
import multiprocessing as mp


def parallel_pool_map(pool: FlexPool, ammount: int, func, *args, **kwargs):
    ids = list(pool.keys())
    pool_size = len(ids)
    chunk_size = pool_size // ammount
    chunks = [ids[i : i + chunk_size] for i in range(0, pool_size, chunk_size)]
    pools = [pool.clients.select(lambda id, _: id in ids) for ids in chunks]
    executor = mp.Pool(ammount)

    executor.map(lambda i, p: p.map(func, rank=i, *args, **kwargs), enumerate(pools))
