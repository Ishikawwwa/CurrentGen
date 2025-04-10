# CurrentGen: Generative for Ocean Currents visualizations

## About the Project

The main idea of this project is to use generative AI methods to get realistic simulations of ocean currents. Basically, a speed field serves as input and a video consisting of several frames through time as output. At the moment, this repository contains a baseline solution of VAE with temporal features(along with TGAN and model architectures experiments) to include time parameter for sequence generation. Future works include using custom initialization, since now it works by generating out of Gaussian Noise, which results in less realistic initial conditions.

Examples of Generated simulations (global physical patterns are obeyed as it can be seen)
![ezgif-6df9c874e5f539](https://github.com/user-attachments/assets/4537dcb4-e750-4324-a491-3f81a35b91f7)

![ezgif-6596504765ca38](https://github.com/user-attachments/assets/20ba7c62-cf27-4f75-b4cf-648b12773fe1)

Also, this repository contains simulations create using FluidSim Library which are less realistic because of perfect initial conditions. (Dipole initialization)

I included it to this page to make it easier to have some expectations of how liquid usually moves in simplified environment.

![ezgif-6878e6cb4b4958](https://github.com/user-attachments/assets/222991b7-e22c-4e0e-88fe-74da8f0a5d94)
