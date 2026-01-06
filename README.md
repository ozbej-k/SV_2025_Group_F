# Collective behaviour 2025/26 project of group F - Model inspired by zebrafish collective behaviour

## Members of group F
- Anja Abramovič | [aanja123](https://github.com/aanja123)
- Ožbej Kresal | [ozbej-k](https://github.com/ozbej-k)
- Matej Rupnik | [mr6231](https://github.com/mr6231)
- Urban Vesel | [ultraviolet99](https://github.com/ultraviolet99)

## Project description
The study used as the starting point for this assignment was [Collignon et al. 2016](#collignon2016). The paper presents a model describing the visual sensory system of zebrafish, introduces a stochastic process based on a probability distribution function to model fish movement depending on their surroundings. The model is then compared to experimental data acquired by recording real life zebrafish and their changes in behaviour when introducing points of interest into the environment. Our goal is to implement the model, validate it by comparing it to the real life experimental data, and then possibly extend it by adding interactive control over positions of points of interest and differently shaped environments to see how the fish react.

## Project plan
### [Milestone: First report - 16. 11. 2025](https://github.com/ozbej-k/Skupinsko-Vedenje-25-26-GroupF/milestone/1):
- Explore existing vision based models and models which explore points of interest
- Set a strategy for reproducing the results in the source paper
- Discuss and plan implementation (select tools and collect any needed data)

### [Milestone: Second report - 7. 12. 2025](https://github.com/ozbej-k/Skupinsko-Vedenje-25-26-GroupF/milestone/2):
- Include improvements from feedback on the previous report
- Fully implement previously discussed implementation
- Describe our implementation in detail (methods used, possible extensions and deviations from original)
- Describe how we will test and compare our implementation with the original

### [Milestone: Third report - 11. 1. 2026](https://github.com/ozbej-k/Skupinsko-Vedenje-25-26-GroupF/milestone/3):
- Include improvements from feedback on the previous report
- Show our results when compared to existing research
- Provide some ideas for future work, improvements and applications of our final model
- Prepare presentation about our work

## Implementation
The implementation of the stochastic vision-based model described in the the source paper can be found in `code/fish_sim`. The model has been extended to run in real-time with a running visualisation of the fish in the environment. 

The model supports bounded homogeneous and heterogeneous environments with spots of interest, where fish percieve their surroundings and gather information based on a 270° field of view and give each percieved stimuli a weight depending on how large they appear on in their field of view.

A screenshot of the simulation running in a heterogeneous environment with 10 fish and 2 spots of interest can be seen on the following image.
<div align="center">
<img src="code/fish_sim/stochastic_sim.png" width="450"/>
</div>

The model defines an orientation and speed probability density function (PDF) which fish sample at every time step. The speed PDF is built empirically from real life recordings of zebrafish. The orientation PDF is a composite of [von Mises](https://en.wikipedia.org/wiki/Von_Mises_distribution) distributions which define the probability of the fish choosing any direction.

The full orientation PDF $f$ is a composite of the following distributions:
- $f_0$ - basic-swimming or wall-following behaviour if close to wall,
- $f_f$ - influence of other percieved fish,
- $f_s$ - influence of percieved spots of interest.

An example of an orientation PDF can be seen on the following image which contains plots of the PDF in cartesian (left) and polar (right) coordinates, where how different stimuli influence the probability of the fish changing its orientation.

<div align="center">
<img src="code/fish_sim/orientation_pdf.png" width="650"/>
</div>

## Interactive simulation
The interactive simulation is located in the `code/simulation` directory.

Running the `sim.py` script will start the simulation, which contains 10 fish and two spots of interest in the bottom left and top right corners.

The interactive simulation supports the following features:
- Add fish by clicking the `+1 Fish` button,
- remove fish by clicking the `-1 Fish` button,
- add spot by clicking the `+1 Spot` button,
- remove spot by clicking the `-1 Spot` button,
- move fish by dragging and dropping them, enabled by toggling the `Move Fish` button, or pressing the `F` key on the keyboard,
- the last clicked / dragged fish will be selected and its orientation PDF will be shown in the circle located at the top right corner as a polar graph, to deselect a fish click anywhere inside the tank while having the `Move Fish` button enabled,
- move spots by dragging and dropping them, enabled by toggling the `Move Spots` button, or pressing the `S` key on the keyboard,
- draw walls by clicking and dragging inside the tank, enabled by toggling the `Draw Walls button`, or pressing the `D` key on the keyboard,
- clear drawn walls by clicking the `Clear Walls` button or pressing the `C` key on the keyboard,
- save drawn walls to `current_tank.png` by pressing the `I` key on the keyboard (to load the drawn tank rename the file to `tank.png`).

## Results
We ran 3 hour long simulations for 4 different environments, which we then compared to real life recorded data of real fish (can be found in `source_paper/Zebrafish_Positions_data`):
- Homogeneous environment with 1 fish
- Homogeneous environment with 10 fish
- Heterogeneous environment with 1 fish and 2 spots
- Heterogeneous environment with 10 fish and 2 spots

The following image contains presence probabilities for a homogeneous environment with 1 fish (top two images) and 10 fish (bottom). The experimental data presence probability can be seen in orange and our simulated presence probability in blue.
<div align="center">
<img src="code/fish_sim/simulations/homo_presence_probability.png" width="550"/>
</div>

The following image contains presence probabilities for a heterogeneous environment with 1 fish (top two images) and 10 fish (bottom). The heterogeneous environment contains two spots of interest marked with red circles. The experimental data presence probability can be seen in orange and our simulated presence probability in blue.
<div align="center">
<img src="code/fish_sim/simulations/hetero_presence_probability.png" width="550"/>
</div>
Our model gives very similar presence probabilities, though they deviate from the experimental data presence probabilities more than the model from the original paper which are almost perfect.
This is likely due to numerical differences in the implementation and is expected to be corrected by tweaking dispersion parameters for the von Mises distributions.

## Future work
We have successfuly implemented the stochastic model and visualised it with a real-time simulation with drag-drop fish control capability. 

Our next step will be to extend the model to allow for non-square environments and control over spots of interest, which will allow us to explore how zebrafish might behave in different environments further.

## References
<a id="collignon2016"></a>
[1] <a href="https://royalsocietypublishing.org/doi/10.1098/rsos.150473">
Collignon, et al. 2016. *A stochastic vision-based model inspired by zebrafish collective behaviour in heterogeneous environments.*
</a>
