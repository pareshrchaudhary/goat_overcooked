# Overcooked

Data structure under [ROOT]
- mapbt
    Codebase for agent training
- mapbt/mapbt/scripts/overcooekd_population
    Core code and command scripts for overcooked
- overcooked_dataset
    Trjactory dataset path
- Overcooked_Population_Data
    Population data, including agent model weights, training scripts, visualized vidoes, config files, etc

The dataset and population data can be generated using commands in mapbt

Always start any script under directory "mapbt/mapbt/scripts" unless otherwise specified
```bash
$ pwd
$ # It should show something like [...]/mapbt/mapbt/scripts
```



## Notation

Replace

- <gpu_id> with the ID of the GPU to run experiment

- <seed_id> with the random seed you want for the experiment

- <layout_name> with the name of the layout for current Overcooked! game




# Pipeline

Now we show an example of training using **MEP** and **MEP + GAMMA** for on the *Counter Circuit* layout



## Step 1: get agent checkpoints

Run the following command to generate agent checkpoints.

```bash
# cd mapbt/mapbt/scritps
bash overcooked_population/scripts/example_mep_counter_circuit/gen_mep_agents.sh <gpu_id>
```



### Customized layout
If you want to work on your own layout, you need to first create the config file for the new layout from existed layouts.

```bash
# at [ROOT]
cd mapbt/mapbt/scripts/overcooked_population/config
bash copy_policy_config_to_new_layout.py --source_layout <any_existed_layout_name> --target_layout <your_new_layout_name> --obs_shape_width <your_layout_width> --obs_shape_height <your_layout_height>
```



### Other population

#### FCP

```bash
# cd mapbt/mapbt/scripts
bash overcooked_population/scripts/example_mep_counter_circuit/gen_sp_agents.sh <gpu_id> <layout_name>
```



## Step 2: get population

See [README](create_population/README.md) under create_population for a compete documentation. Here we continue out example of the MEP population.



### Find you checkpoints

First you will be able to find all training logs and models under this directory. If you run the example scripts for multiple times, find the latest **runx** to replace **run1** here.

```bash
# at [ROOT]
mapbt/mapbt/scripts/results/Overcooked/counter_circuit_o_1order/mep/mep_agents-S1/run1
```



### Create checkpoint path config file

Now create a new subfolder *counter_circuit_o_1order_mep* under the *Overcooked_Population_Data* directory to store all population data.

```bash
# cd overcooekd_pop_data
mkdir counter_circuit_o_1order_mep
cd counter_circuit_o_1order_mep
```

Now create an *index_config.json* file under *counter_circuit_o_1order_mep* to indicate the path to the models and their training logs. A detailed description of the *index_config.json" can be found [here]().

We provide an example file and you can simply copy that. Be sure to replace **run1** to your own **runx**.

```bash
# at [ROOT]
cp mapbt/mapbt/scripts/overcooked_population/scr
ipts/example_mep_counter_circuit/extract_config.json overcooekd_pop_data/counter_circuit_o_1order_mep/index_config.json
```



### Construct the population

Then extract three checkpoints (init/mid/final) from all checkpoints of one agent. This scripts automatically visualize all agents, so it takes a few minitus for the population construction to finish. You can then check the produced gif files to test if your previous steps are coorect.

```bash
# cd mapbt/mapbt/scripts
python overcooked_population/create_population/eval_extract_render.py counter_circuit_o_1order_mep
```

You should be able to find gifs file like
```bash
# at [ROOT]
overcooked_population/counter_circuit_o_1order_mep/visualization/mep1_final/reward_160.gif
```


## Step -1: get evaluation agents

Actually, we need held-out agents or human proxy model for evaluation. You need some further steps to get them. We will introduce how to do that in thie section. Fortunately, we already have those models and you can directly use them. For example, you can directly copy those files

```bash
bc_train.pt
bc_test.pt
sp9_init_actor.pt
sp9_mid_actor.pt
sp9_final_actor.pt
sp10_init_actor.pt
sp10_mid_actor.pt
sp10_final_actor.pt
```

from *overcooekd_pop_data/counter_circuit_o_1order_fcp/models* to *overcooekd_pop_data/counter_circuit_o_1order_mep/models*.


### How to get human proxy models (bc_train/bc_test) from scratch

TODO



### How to get held-out agents (sp9, sp10) from scratch

TODO




## Step 2.5: train MEP Cooperators

Now you already have the MEP population to train a normal MEP Cooperator. Use our example scripts is fine, but maybe you also want to costomize it to use wandb (the example script disables wandb).

First create the population config file for zero-shot coordination. (We already create one so just copy+paste.)

```bash
# at [ROOT]
cp mapbt/mapbt/scripts/overcooked_population/scripts/example_mep_counter_circuit/zsc_config.yml Overcooked_Population_Data/counter_circuit_o_1order_mep/zsc_config.yml 
```

Then run the *zsc_mep_agent.sh* command.

```bash
# cd mapbt/mapbt/scripts
bash overcooked_population/scripts/example_mep_counter_circuit/zsc_mep_agent.sh <gpu-id> <seed-id>
```



## Step 3: collect dataset

There is one more step for GAMMA to train a generative model of the population. The first step to do so is to create a dataset of the population.

You will find a VAE training population config file is automatically generated here.

```bash
# at [ROOT]
Overcooked_Population_Data/counter_circuit_o_1order_mep/pop_for_vae_config.yml
```

While this one is already sufficient to train the VAE, we recommand you also add some held-out agents to also have some test accuracy for the VAE model. We already create one so you can also do copy-paste.

```bash
# at [ROOT]
cp mapbt/mapbt/scripts/overcooked_population/scripts/example_mep_counter_circuit/pop_for_vae_config.yml Overcooked_Population_Data/counter_circuit_o_1order_mep/pop_for_vae_config.yml 
```

Now you can use the following command to generate the training datset and test dataset for VAE.

```bash
# cd mapbt/mapbt/scripts
bash overcooked_population/scripts/example_mep_counter_circuit/create_dataset.sh <gpu-id>
```

And you should be able to find a dataset file at:

```bash
# at [ROOT]
overcooked_dataset/counter_circuit_o_1order_mep/?
```



## Step 4: train generative policies

After the dataset is generated, now it is time to train the generative model using the following command.

```bash
# cd mapbt/mapbt/scripts
bash overcooked_population/scripts/example_mep_counter_circuit/train_vae.sh <gpu-id>
```

To find the trained VAE models, check this folder.

```bash
# at [ROOT]
mapbt/mapbt/scripts/results/Overcooked/diverse_counter_circuit_6x5/population/train_vae.sh/run1/models # replace run1 with your correct runx
```

You will find different checkpoints corresponding to different saving metric. Here we pick this one.

```bash
encoder_best_log_likelihood_kl_32.pt
decoder_best_log_likelihood_kl_32.pt
```

This VAE is the one with the minimum log likelihood loss under the constraint that the KL divergence bettwe the prio and posterior distribution is less than 32.

Move those two checkpoints under this folder
```bash
# at [ROOT]
Overcooked_Population_Data/diverse_counter_circuit_6x5_mep/vae_models/best_logp_kl_32
```
and rename them as
```bash
# at [ROOT]
Overcooked_Population_Data/diverse_counter_circuit_6x5_mep/vae_models/best_logp_kl_32/encoder.pt
Overcooked_Population_Data/diverse_counter_circuit_6x5_mep/vae_models/best_logp_kl_32/decoder.pt
```



## Step 5: train GAMMA Cooperators


Now we have everything for Cooperator training! Use the population config file here. (You probably already have done this step for vanilla MEP.)

```bash
# at [ROOT]
cp mapbt/mapbt/scripts/overcooked_population/scripts/example_mep_counter_circuit/zsc_config.yml Overcooked_Population_Data/counter_circuit_o_1order_mep/zsc_config.yml 
```

And run the following command. Start training!

```bash
# cd mapbt/mapbt/scripts
overcooked_population/scripts/example_mep_counter_circuit/train_mep_kl32.sh <gpu-id>
```
