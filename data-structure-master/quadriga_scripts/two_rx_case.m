close all
clear all

% specify path to scr folder
% addpath('path/to/quadriga_src')


s = qd_simulation_parameters;                                               % Create quadriga layout
s.center_frequency = 2.53e9;                                                % 2.53 GHz carrier carrier freqency
s.sample_density = 4;                                                       % 4 samples per half-wavelenght
s.use_absolute_delays = 0;                                                  % use NLOS path

l = qd_layout(s);
l.no_rx = 2;
l.tx_array = qd_arrayant('3gpp-3d', 1, 1, s.center_frequency, 1);           % create array with Ain = 1; Bin = 1; .Cin = 2.53; .Din = 1;
l.rx_array = l.tx_array;                                                    % use same antennas for both tx and rx
l.tx_position(3) = 25;

UMan = '3GPP_3D_UMa_NLOS';

l.track = qd_track('linear', 20, pi/2);                                     % Linear track, 20 m length
l.track.initial_position = [10; 0; 1.5];                                    % Start east running north
l.track.interpolate_positions(s.samples_per_meter);
l.track.segment_index = [1];                                                % segments
l.track.scenario = {UMan};                                                  % scenario


l.visualize;

interpolate_positions(l.track, s.samples_per_meter);
compute_directions(l.track);                                                % align antenna direction with track

p = l.init_builder;                                                         % create channel builders
p.gen_ssf_parameters;                                                       % generate small-scale fading

s.use_spherical_waves = 1;
c = get_channels(p);                                                        % generate channel coefficients
cn = merge(c);

h = cn(1, 1).fr(100e6, 512);                                                % freq.-domain channel

save("-hdf5", ['data_two_rx_case'],'h')