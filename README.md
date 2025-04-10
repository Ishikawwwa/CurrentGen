# CurrentGen: Generative for Ocean Currents visualizations

## About the Project

The main idea of this project is to use generative AI methods to get realistic simulations of ocean currents. Basically, a speed field serves as input and a video consisting of several frames through time as output. At the moment, this repository contains a baseline solution of VAE with temporal features(along with TGAN experiments) to include time parameter for sequence generation. Future works include using custom initialization, i.e. Dipole initialization, since now it works by generating out of Gaussian Noise, which results in less realistic initial conditions.
