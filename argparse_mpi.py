import argparse
from pathlib import Path
from mpi4py import MPI

import numpy as np
import h5py

###
# Create command line arg parser
###

def create_parser():
    '''
    Create a command line argument parser that grabs the number of nodes
        and the parameter text file. Allow for verbosity

    Args:
        ...
    Returns:
        ...
    '''

    parser = argparse.ArgumentParser(
        description="Testing mpi argparse script")

    parser.add_argument("h5fname", help="File name of test hdf5 file", type=str)

    parser.add_argument("iterations", help='Numbers to print', type=int)

    parser.add_argument('-v', '--verbose', help='Print info along the way', 
                        action='store_true')

    return parser


def main():
    '''
    Test using argparse and MPI
    '''

    comm = MPI.COMM_WORLD
    rank = comm.Get_rank()
    size = comm.Get_size()

    rank_idstr = f"Rank {rank:.0f}"

    if rank == 0:
        parser = create_parser()
        args = parser.parse_args()
        if args.verbose:
            print(f"--- {rank_idstr} : Args parsed and created ! ---")
    else:
        args = None


    args = comm.bcast(args, root=0)

    if args.verbose and rank == 0:
        print(f"--- {rank_idstr} : Args have been broadcasted! ---")

    print(comm.Get_rank(), args.iterations)


    with h5py.File(args.h5fname, 'w', driver='mpio', comm=comm) as fObj:
        if args.verbose and rank == 0:
            print(f"--- {rank_idstr} : File object created!")
        dset = fObj.create_dataset('test', (size,), dtype='i')
        dset[rank] = rank


        

if __name__=="__main__":
    main()

