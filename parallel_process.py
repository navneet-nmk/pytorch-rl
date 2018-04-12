"""
This script spawns 2 processes who will each setup the distributed environments ,
initialize the process group and finally execute the given run function
"""


import os
import torch
import torch.distributed as dist
from torch.multiprocessing import Process


""" Blocking point to point communication. """


def run_p2p(rank, size):
    tensor = torch.zeros(1)
    if rank == 0:
        tensor += 1
        # Send the tensor to process 1
        dist.send(tensor=tensor, dst=1)
    else:
        # Receive tensor from process 0
        dist.recv(tensor=tensor, src=0)

    print('Rank ', rank, ' has data ', tensor[0])


""" Non Blocking point to point communication"""


def run_non_p2p(rank, size):
    tensor = torch.zeros(1)
    req = None
    if rank == 0:
        tensor += 1
        # Send the tensor to process 1
        req = dist.isend(tensor=tensor, dst=1)
        print('Rank 0 started sending')
    else:
        # Receive tensor from process 0
        req = dist.irecv(tensor=tensor, src=0)
        print('Rank 1 started receiving')

    # We should not modify the sent tensor nor access the received tensor before req.wait()
    req.wait()
    print('Rank ', rank, ' has data ', tensor[0])


""" Collective Communication - All reduce example """


def run_all_reduce(rank, size):
    """
    As opposed to point to point communication, collective communication
    allow for communication patterns across all processes in a group.
    :param rank:
    :param size:
    :return:
    """

    """ Simple point to point communication """
    group = dist.new_group([0, 1])
    tensor = torch.ones(1)
    dist.all_reduce(tensor, op=dist.reduce_op.SUM, group=group)
    print('Rank ', rank, ' has data ', tensor[0])



def run(rank, size):
    """
    Distributed function to be run
    :param rank:
    :param size:
    :return:
    """

    pass


def init_processes(rank, size, fn, backend='tcp'):
    """
    Initialize the distributed environment
    :param rank:
    :param size:
    :param fn:
    :param backend:
    :return:
    """

    os.environ['MASTER_ADDR'] = '127.0.0.1'
    os.environ['MASTER_PORT'] = '29500'
    # This method essentially allows processes to communicate with each other by sharing their positions
    dist.init_process_group(backend=backend, rank=rank, world_size=size)
    fn(rank, size)


if __name__ == "__main__":
    size = 2
    processes = []
    for rank in range(size):
        p = Process(target=init_processes, args=(rank, size, run))
        p.start()
        processes.append(p)

    for p in processes:
        p.join()

