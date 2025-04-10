# CurrentGen: Generative for Ocean Currents visualizations

## About the Project

The main idea of this project is to use generative AI methods to simulate Navier-Stocks fluid modeling. Basically, a speed field serves as input and a video consisting of several frames through time as output. At the moment, this repository contains a baseline solution of VAE with temporal features to include time parameter for sequence generation. Nearest future works plan contains trying out GANs and Diffusion models and training on real-world data instead of randomly-initialized simulation. Current visualization is also already finished and made with FluidSim library that utilizes Navier-Stocks to calculate the approximate motion of water. So, at the moment the project is half way there, but I would like to play with different approaches and measure inference speed, since this solution is intended as a way to make fast visualizations instead of complex mathematical modeling.
