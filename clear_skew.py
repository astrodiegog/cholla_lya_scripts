import argparse
from pathlib import Path

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
        description="Reset skewer file back to default Cholla outputs")

    parser.add_argument("skewfname", help='Cholla skewer output file name', type=str)

    parser.add_argument('-v', '--verbose', help='Print info along the way', 
                        action='store_true')

    return parser



def main():
    '''
    Remove any datasets not included in Cholla skewer outputs
    '''

    # Create parser
    parser = create_parser()

    # Save args
    args = parser.parse_args()

    if args.verbose:
        print("we're verbose in this mf !")
        print(f"--- We are looking at skewer file : {args.skewfname} ---")

    precision = np.float64

    # Convert argument input to Path() & get its absolute path
    skewer_fPath = Path(args.skewfname).resolve()
    assert skewer_fPath.is_file()

    # keep default Cholla skewer outputs from datasets
    keys2keep = ['HI_density', 'HeII_density', 'density', 'los_velocity', 'temperature', 'vel_Hubble']

    with h5py.File(skewer_fPath, 'r+') as fObj:
        for i_skew in fObj.keys():
            if args.verbose:
                print('--- ', i_skew)
            for attr in dict(fObj[i_skew].attrs).keys():
                if args.verbose:
                    print('--- \t', attr)
                fObj[i_skew].attrs.pop(attr)

            for data_key in fObj[i_skew].keys():
                if args.verbose:
                    print('--- \t', data_key)
                if data_key not in keys2keep:
                    del fObj[i_skew][data_key]

    print(f"--- Done cleaning up file : {skewer_fPath} ---")

        

if __name__=="__main__":
    main()

