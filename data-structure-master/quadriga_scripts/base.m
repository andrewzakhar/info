close all
clear all

addpath('D:\ChannelGenerator\quadriga_src')


s = qd_simulation_parameters;                                               % Create quadriga layout
s.center_frequency = 2.53e9;                                                % 2.53 GHz carrier carrier freqency
s.sample_density = 4;                                                       % 4 samples per half-wavelenght
s.use_absolute_delays = 0;                                                  % use NLOS path

l = qd_layout( s );

antTx = qd_arrayant('3gpp-mmw', 1, 1);  % generate antenna
l.no_tx = 1;                               % number of BSs
l.tx_array = copy(antTx);
l.tx_position(:, 1) = [-50 ; 0 ; 25];      % position of BS1

antRx = qd_arrayant('3gpp-mmw', 1, 1);  % generate antenna
l.no_rx = 1;
l.rx_position(3) = 25
l.rx_array = copy(antRx);

UMan2 = '3GPP_3D_UMa_NLOS';

l.track = qd_track('linear', 20, pi/2);                                     % Linear track, 20 m length
l.track.initial_position = [0; 0; 25];                                      % Linear track, 20 m length
l.track.interpolate_positions(s.samples_per_meter);
l.track.scenario = {UMan2};                                                 % scenario


l.visualize;

interpolate_positions(l.track, s.samples_per_meter);
compute_directions(l.track);                                                % align antenna direction with track

p = l.init_builder;                                                         % create channel builders
p.gen_ssf_parameters;                                                       % generate small-scale fading

s.use_spherical_waves = 1;
c = get_channels(p);                                                        % generate channel coefficients
cn = merge(c);
h = cn(1, 1).fr(100e6, 512);                                                  % freq.-domain channel

save("-hdf5", ['data_base'],'h')
