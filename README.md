# Drag-Based Editing of Images through Energy Based Guidance | [Project Page]([https://anisundar18.github.io/aeroblade_mod/](https://anisundar18.github.io/DragGuidance/))


## Overview
Drag-based image editing is a style of image editing where the user can "drag" any points of the image to precisely reach target points. Successful drag-based editing involves moving the object as well as preserving the other details in the original image. This repository contains the code pertaining to an exploratory project that relies on using feature correspondences in large-scale diffusion models in order to perform Drag-based image editing. For a detailed write-up on the project, please refer to the Project Page linked.

## Performing Drag Based Edits
The UI has not been set up yet, however, the code can be run using the `run.sh` script. As of now, the handle and target points unfortunately have to be hard-coded. Furthermore, the guidance is based on the UNet returning feature maps, therefore the original `unet_2d_condition.py` has to be replaced with the one provided in the repository.


