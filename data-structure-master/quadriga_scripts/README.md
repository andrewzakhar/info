Before start, please download the quadriga src and specify the path in script.

In these simulations I am using `TR 38.901 v16.1.0`. \
I am not sure that it's fully correct simulation.


The structure of simulation

1. Define simulation parameters (frequency for center carrier etc.)
2. Define layout (number of rx/tx, type of rx/tx)
3. Define track type for rx (set scenario, type of movement)
4. Get channel and transform it to frequency domain

For more information check the official documentation. PDF file contains almost all info.

To actually run the scripts I am using an octave and then saving the data into `hdf5` format.
The `hdf5` files are easily to read in python or any other language.

Scripts description:
1. `two_tx_case.m` - we have 1 rx antenna and two tx antennas
2. `two_rx_case.m` - we have 2 rx and 1 tx
3. `base.m` - we have 1 rx and 1 tx

