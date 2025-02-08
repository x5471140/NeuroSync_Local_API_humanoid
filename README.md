# NeuroSync Open Source Audio2Face Local API

## NEW : [Train your own model](https://github.com/AnimaVR/NeuroSync_Trainer_Lite)

## 01/01/25 Model + code updates

The final version of the 0.01 model has been added. V0.02 is due february with a new architecture.

You MUST update all .py as the old ones wont work with the latest model (many changes involved in the model over this research period!)

This version is the best so far but struggles with dimensions at the furthest points from the mouth, some dimensions are zero'd out to keep things smooth.

## Talk to a NeuroSync prototype live on Twitch : [Visit Mai](https://www.twitch.tv/mai_anima_ai)

## Overview

The **NeuroSync Local API** allows you to host the audio-to-face blendshape transformer model locally. This API processes audio data and outputs facial blendshape coefficients, which can be streamed directly to Unreal Engine using the **NeuroSync Player** and LiveLink.

### Features:
- Host the model locally for full control
- Process audio files and generate facial blendshapes

## NeuroSync Model

To generate the blendshapes, you can:

- [Download the model from Hugging Face](https://huggingface.co/AnimaVR/NEUROSYNC_Audio_To_Face_Blendshape)
- [Apply for Alpha API access](https://neurosync.info) if you prefer not to host the model locally.

## Player Requirement

To stream the generated blendshapes into Unreal Engine, you will need the **NeuroSync Player**. The Player allows for real-time integration with Unreal Engine via LiveLink. 

You can find the NeuroSync Player and instructions on setting it up here:

- [NeuroSync Player GitHub Repository](https://github.com/AnimaVR/NeuroSync_Player)

Visit [neurosync.info](https://neurosync.info) for more details and to sign up for alpha access if you wish to use the non-local API option.
