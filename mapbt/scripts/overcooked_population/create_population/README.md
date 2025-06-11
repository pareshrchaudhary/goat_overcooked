# Saved data for each population

## Data Structure for [pop_data]

The top folder is created for each population "{layout_name}_{pop_name}", for example:
- diverse_shaped_pop
- circuit_fcp

Under each population, saving
- index_config.json
    - layout
    - local_policies:
        - "sp1":
            - script: script A (the script to get this agent)
            - seed: 1 (the seed used in the script)
            - log_file (the file that contains the training statistics, e.g., reward during training time)
            - model_dir (the folder containing checkpoint files)
            - held_out ("True" means this agent is used for out-of-population evaluation)
        - "mep1":
            - ...
    - web_policies: TODO
- scripts
    - script A
    - script B
    - ...
- models
    - run1_1.pt
    - run1_2.pt
    - ...
- visualization
    - run1_1/reward_R.gif
    - ...
- pop_for_vae_config.yml
- vae_models
    - TODO


## Create population data from raw agent checkpoints

Creating the "index_config.json" file. You will need to indicate the *log_file* so our extraction tool can get the rewards of different checkpoints, and the *model_dir* so we can find the checkpoint files. See an example [here](../pop_data/asymmetric_advantages_fcp/index_config.json)

Move all scripts into the "scripts" subdir

Run

```bash
# cd mapbt/mapbt/scripts
python overcooked_population/create_population/eval_extract_render.py [population_name]
```

to extract populaiton agents into the [models] folder and create gifs of all agents in the [visualization] folder
