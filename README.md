<div id="top"></div>
<!--
*** Thanks for checking out the Best-README-Template. If you have a suggestion
*** that would make this better, please fork the repo and create a pull request
*** or simply open an issue with the tag "enhancement".
*** Don't forget to give the project a star!
*** Thanks again! Now go create something AMAZING! :D 
-->



<!-- PROJECT SHIELDS -->
<!--
*** I'm using markdown "reference style" links for readability.
*** Reference links are enclosed in brackets [ ] instead of parentheses ( ).
*** See the bottom of this document for the declaration of the reference variables
*** for contributors-url, forks-url, etc. This is an optional, concise syntax you may use.
*** https://www.markdownguide.org/basic-syntax/#reference-style-links
-->
[![Contributors][contributors-shield]][contributors-url]
[![Forks][forks-shield]][forks-url]
[![Stargazers][stars-shield]][stars-url]
[![Issues][issues-shield]][issues-url]
[![MIT License][license-shield]][license-url]



<!-- PROJECT LOGO -->
<br />
<div align="center">
  <a href="https://github.com/albertemc2stein/PINN">
    <img src="https://cdn-icons-png.flaticon.com/512/2103/2103633.png" alt="Logo" width="80" height="80">
  </a>

  <h3 align="center">PINN</h3>

  <p align="center">
    A framework for solving differential equations with neural networks.
    <br />
    <a href="https://github.com/albertemc2stein/PINN"><strong>Explore the docs »</strong></a>
    <br />
    <br />
    <a href="https://github.com/albertemc2stein/PINN">View Demo</a>
    ·
    <a href="https://github.com/albertemc2stein/PINN/issues">Report Bug</a>
    ·
    <a href="https://github.com/albertemc2stein/PINN/issues">Request Feature</a>
  </p>
</div>



<!-- TABLE OF CONTENTS -->
<details>
  <summary>Table of Contents</summary>
  <ol>
    <li>
      <a href="#about-the-project">About The Project</a>
      <ul>
        <li><a href="#built-with">Built With</a></li>
      </ul>
    </li>
    <li><a href="#usage">Usage</a></li>
    <li><a href="#roadmap">Roadmap</a></li>
    <li><a href="#contributing">Contributing</a></li>
    <li><a href="#license">License</a></li>
    <li><a href="#contact">Contact</a></li>
  </ol>
</details>



<!-- ABOUT THE PROJECT -->
## About The Project

[![Product Name Screen Shot][product-screenshot]](https://en.wikipedia.org/wiki/Physics-informed_neural_networks)

This project makes use of the fact that neural networks are essentially nothing more than universal
function approximators. In essence, the neural network is trained upon minimizing a loss function
that measures the deviation from a solution to the given differential equation (and additional constraints).

Basic approach:
* Create a new boundary value problem
* Add constraints and the regions hey are acting on
* Tell the network which differentials it needs to calculate 
* Train the network

This is obviously only a very basic description of one of the many ways to utilize this approach.
The many examples provided with this package show the abilities (and limits!) of this framework.

<p align="right">(<a href="#top">back to top</a>)</p>



### Built With
* [TensorFlow](https://www.tensorflow.org/) (Neural network support)
* [NumPy](https://numpy.org/) (Efficient sampling)
* [Matplotlib](https://matplotlib.org/) (Visualization)

<p align="right">(<a href="#top">back to top</a>)</p>



<!-- GETTING STARTED -->
## Getting Started

This is an example of how you set up this project locally.

1. Create a new virtual environment
   ```
   python3 -m venv /path/to/new/virtual/environment
   ```
   If this does not work check that *venv* is installed and execute
   ```
   pip3 install virtualenv
   ```
   if neccessary.


2. Activate the new environment with
   ```
   source /path/to/new/virtual/environment/bin/activate
   ```

3. Clone the repository
   ```
   git clone https://github.com/AlbertEMC2Stein/PINN.git
   ```

4. Install the PDESolver package with
   ```
   pip3 install -e .
   ```
   from within the PINN directory (this might take a minute).


5. Use the package for your own purposes or try out some of the example scripts with
   ```
   python3 scripts/script_name.py
   ```
   To see all available scripts use ``ls scripts`` first

<p align="right">(<a href="#top">back to top</a>)</p>



<!-- USAGE EXAMPLES -->
## Usage

_For usage examples, please refer to the [provided scripts](https://github.com/AlbertEMC2Stein/PINN/tree/main/scripts)_

<p align="right">(<a href="#top">back to top</a>)</p>



<!-- ROADMAP -->
## Roadmap

- [x] Add arbitrary dimension support
- [x] Add different sampling methods 
- [ ] Add more region types (spherical, cylindrical, etc.)
- [ ] Add better visulization support
    - [ ] 1D
    - [ ] 2D

See the [open issues](https://github.com/albertemc2stein/PINN/issues) for a full list of proposed features (and known issues).

<p align="right">(<a href="#top">back to top</a>)</p>



<!-- CONTRIBUTING -->
## Contributing

If you have a suggestion that would make this better, please fork the repo and create a pull request. You can also simply open an issue with the tag "enhancement".
Don't forget to give the project a star! Thanks again!

1. Fork the Project
2. Create your Feature Branch (`git checkout -b feature/NewFeature`)
3. Commit your Changes (`git commit -m 'Add some NewFeature'`)
4. Push to the Branch (`git push origin feature/NewFeature`)
5. Open a Pull Request

<p align="right">(<a href="#top">back to top</a>)</p>



<!-- LICENSE -->
## License

Distributed under the MIT License. See `LICENSE.txt` for more information.

<p align="right">(<a href="#top">back to top</a>)</p>



<!-- CONTACT -->
## Contact

Tim Prokosch - prokosch@rhrk.uni-kl.de

Project Link: [https://github.com/albertemc2stein/PINN](https://github.com/albertemc2stein/PINN)

<p align="right">(<a href="#top">back to top</a>)</p>



<!-- ACKNOWLEDGMENTS -->
## Acknowledgments

* [Implementation for Burgers equation](https://github.com/janblechschmidt/PDEsByNNs/blob/main/PINN_Solver.ipynb)
* [Paper by Maziar Raissi](https://www.sciencedirect.com/science/article/pii/S0021999118307125)


<p align="right">(<a href="#top">back to top</a>)</p>



<!-- MARKDOWN LINKS & IMAGES -->
<!-- https://www.markdownguide.org/basic-syntax/#reference-style-links -->
[contributors-shield]: https://img.shields.io/github/contributors/albertemc2stein/PINN.svg?style=for-the-badge
[contributors-url]: https://github.com/albertemc2stein/PINN/graphs/contributors
[forks-shield]: https://img.shields.io/github/forks/albertemc2stein/PINN.svg?style=for-the-badge
[forks-url]: https://github.com/albertemc2stein/PINN/network/members
[stars-shield]: https://img.shields.io/github/stars/albertemc2stein/PINN.svg?style=for-the-badge
[stars-url]: https://github.com/albertemc2stein/PINN/stargazers
[issues-shield]: https://img.shields.io/github/issues/albertemc2stein/PINN.svg?style=for-the-badge
[issues-url]: https://github.com/albertemc2stein/PINN/issues
[license-shield]: https://img.shields.io/github/license/albertemc2stein/PINN.svg?style=for-the-badge
[license-url]: https://github.com/albertemc2stein/PINN/blob/master/LICENSE.txt
[product-screenshot]: https://www.researchgate.net/profile/Zhen-Li-105/publication/335990167/figure/fig1/AS:806502679982080@1569296631121/Schematic-of-a-physics-informed-neural-network-PINN-where-the-loss-function-of-PINN.png