import numpy as np

import torch
import torch.nn as nn
import torch.nn.functional as F

from .util import init

class Flatten(nn.Module):
    def forward(self, x):
        return x.view(x.size(0), -1)

class CNNLayer(nn.Module):
    def __init__(self, obs_shape, hidden_size, use_orthogonal, activation_id, kernel_size=3, stride=1):
        super(CNNLayer, self).__init__()

        active_func = [nn.Tanh(), nn.ReLU(), nn.LeakyReLU(), nn.ELU()][activation_id]
        init_method = [nn.init.xavier_uniform_, nn.init.orthogonal_][use_orthogonal]
        gain = nn.init.calculate_gain(['tanh', 'relu', 'leaky_relu', 'leaky_relu'][activation_id])

        def init_(m):
            return init(m, init_method, lambda x: nn.init.constant_(x, 0), gain=gain)

        input_channel = 20 # obs_shape[2]
        input_width = obs_shape[0]
        input_height = obs_shape[1]

        self.cnn = nn.Sequential(
            init_(nn.Conv2d(in_channels=input_channel, out_channels=hidden_size//2, kernel_size=kernel_size, stride=stride)), active_func,
            Flatten(),
            init_(nn.Linear(hidden_size//2 * (input_width-kernel_size+stride) * (input_height-kernel_size+stride), hidden_size)), active_func,
            init_(nn.Linear(hidden_size, hidden_size)), active_func)

    def forward(self, x):
        x = x / 255.0
        OLD_CHANNELS = [
            "p0_loc",
            "p1_loc",
            "p0_north",
            "p0_south",
            "p0_east",
            "p0_west",
            "p1_north",
            "p1_south",
            "p1_east",
            "p1_west",
            # for i, player in enumerate(overcooked_state.players):
            #     player_orientation_idx = Direction.DIRECTION_TO_INDEX[
            #         player.orientation
            #     ]
            #     state_mask_dict["player_{}_loc".format(i)] = make_layer(
            #         player.position, 1
            #     )
            #     state_mask_dict[
            #         "player_{}_orientation_{}".format(
            #             i, player_orientation_idx
            #         )
            #     ] = make_layer(player.position, 1)
            # def make_layer(position, value):
            #     layer = np.zeros(self.shape)
            #     layer[position] = value
            #     return layer
            # ALL_DIRECTIONS = INDEX_TO_DIRECTION = [NORTH, SOUTH, EAST, WEST]
            # DIRECTION_TO_INDEX = {a: i for i, a in enumerate(INDEX_TO_DIRECTION)}
            "pot_loc",
            # for loc in self.get_pot_locations():
            #     state_mask_dict["pot_loc"][loc] = 1
            # def get_pot_locations(self):
            #     return list(self.terrain_pos_dict["P"])
            "counter_loc",
            # for loc in self.get_counter_locations():
            #     state_mask_dict["counter_loc"][loc] = 1
            # def get_counter_locations(self):
            #     return list(self.terrain_pos_dict["X"])
            "onion_disp_loc",
            # for loc in self.get_onion_dispenser_locations():
            #     state_mask_dict["onion_disp_loc"][loc] = 1
            # def get_onion_dispenser_locations(self):
            #     return list(self.terrain_pos_dict["O"])
            "tomato_disp_loc",
            # for loc in self.get_tomato_dispenser_locations():
            #     state_mask_dict["tomato_disp_loc"][loc] = 1
            # def get_tomato_dispenser_locations(self):
            #     return list(self.terrain_pos_dict["T"])
            "dish_disp_loc",
            # for loc in self.get_dish_dispenser_locations():
            #     state_mask_dict["dish_disp_loc"][loc] = 1
            # def get_dish_dispenser_locations(self):
            #     return list(self.terrain_pos_dict["D"])
            "serve_loc",
            # for loc in self.get_serving_locations():
            #     state_mask_dict["serve_loc"][loc] = 1
            # def get_serving_locations(self):
            #     return list(self.terrain_pos_dict["S"])
            "onions_in_pot",
            "tomatoes_in_pot",
            # Count the number of onions in the pot when the pot has not been started
            # if obj.position in self.get_pot_locations():
            #     if obj.is_idle:
            #         # onions_in_pot and tomatoes_in_pot are used when the soup is idling, and ingredients could still be added
            #         state_mask_dict["onions_in_pot"] += make_layer(
            #             obj.position, ingredients_dict["onion"]
            #         )
            "onions_in_soup",
            "tomatoes_in_soup",
            # Count the number of onions in the soup when the pot is cooking/or when the soup is not in the pot
            "soup_cook_time_remaining",
            "soup_done",
            # obj._cooking_tick = -1 if idle else #seconds since started
            # if obj.position in self.get_pot_locations():
            #     if obj.is_idle:
            #     else:
            #         state_mask_dict["onions_in_soup"] += make_layer(
            #             obj.position, ingredients_dict["onion"]
            #         )
            #         state_mask_dict["tomatoes_in_soup"] += make_layer(
            #             obj.position, ingredients_dict["tomato"]
            #         )
            #         state_mask_dict[
            #             "soup_cook_time_remaining"
            #         ] += make_layer(
            #             obj.position, obj.cook_time - obj._cooking_tick
            #         )
            #         if obj.is_ready:
            #             state_mask_dict["soup_done"] += make_layer(
            #                 obj.position, 1
            #             )

            # else:
            #     # If player soup is not in a pot, treat it like a soup that is cooked with remaining time 0
            #     state_mask_dict["onions_in_soup"] += make_layer(
            #         obj.position, ingredients_dict["onion"]
            #     )
            #     state_mask_dict["tomatoes_in_soup"] += make_layer(
            #         obj.position, ingredients_dict["tomato"]
            #     )
            #     state_mask_dict["soup_done"] += make_layer(
            #         obj.position, 1
            #     )
            "dishes",
            "onions",
            "tomatoes",
            # elif obj.name == "dish":
            #     state_mask_dict["dishes"] += make_layer(obj.position, 1)
            # elif obj.name == "onion":
            #     state_mask_dict["onions"] += make_layer(obj.position, 1)
            # elif obj.name == "tomato":
            #     state_mask_dict["tomatoes"] += make_layer(obj.position, 1)
            "urgency",
            # if horizon - overcooked_state.timestep < 40:
                # state_mask_dict["urgency"] = np.ones(self.shape)
        ]
        NEW_CHANNELS = [
            "p0_loc",
            "p1_loc",
            "p0_north",
            "p0_south",
            "p0_east",
            "p0_west",
            "p1_north",
            "p1_south",
            "p1_east",
            "p1_west",
            "pot_loc",
            "counter_loc",
            "onion_disp_loc",
            "dish_disp_loc",
            "serve_loc",
            # AIR = 0
            # POT = 1
            # COUNTER = 2
            # ONION_SOURCE = 3
            # DISH_SOURCE = 4
            # SERVING = 5
            "is soup + is pot + num_onions",
            "is soup + is pot + cooking_tick",
            "is soup + not pot",
            # obj._cooking_tick = -1 if it has not started cooking
            # obj._cooking_tick = passed time for cooking
            # if obj.name == SOUP:
            #     if self.get_terrain(pos) == POT:
            #         # if obj._cooking_tick < 0:
            #         #     obs[pos, shift + 6] = obj.num_onions
            #         #     obs[pos, shift + 7] = obj.num_tomatoes
            #         # else:
            #         #     obs[pos, shift + 8] = obj.num_onions
            #         #     obs[pos, shift + 9] = obj.num_tomatoes
            #         #     obs[pos, shift + 10] = self.get_time(obj) - obj._cooking_tick
            #         #     if self.is_ready(obj):
            #         #         obs[pos, shift + 11] = 1
            #         obs[pos, shift + 5] = obj.num_onions
            #         if obj._cooking_tick < 0:
            #             obs[pos, shift + 6] = 0
            #         else:
            #             obs[pos, shift + 6] = obj._cooking_tick
            #     else:
            #         # obs[pos, shift + 8] = obj.num_onions
            #         # obs[pos, shift + 9] = obj.num_tomatoes
            #         # obs[pos, shift + 10] = 0
            #         # obs[pos, shift + 11] = 1
            #         obs[pos, shift + 7] = 1
            #     # print("SOUP at", pos)
            "dishes",
            # elif obj.name == DISH:
            #     # obs[pos, shift + 12] = 1
            #     obs[pos, shift + 8] = 1
            #     # print("DISH at", pos)
            "onions",
            # elif obj.name == ONION:
            #     # obs[pos, shift + 13] = 1
            #     obs[pos, shift + 9] = 1
            #     # print("ONION at", pos)
            # # elif obj.name == TOMATO:
            #     # obs[pos, shift + 14] = 1
        ]
        COOK_TIME = 20
        CALCULATION = {
            "is soup + is pot + num_onions": (lambda x: \
                x["pot_loc"] * (x["onions_in_pot"] + x["onions_in_soup"])
            ),
            "is soup + is pot + cooking_tick": (lambda x: \
                x["pot_loc"] * (x["soup_done"] * COOK_TIME + (1 - x["soup_done"]) * (x["onions_in_soup"] + x["tomatoes_in_soup"] > 0) * (COOK_TIME - x["soup_cook_time_remaining"]))
            ),
            "is soup + not pot": (lambda x: \
                x["soup_done"] * (1 - x["pot_loc"])
            )
        }
        x = x.movedim(-1, -3)
        x = {
            k: x[:, i]
            for i, k in enumerate(OLD_CHANNELS)
        }
        assert torch.all(torch.isclose(x["soup_done"], torch.zeros_like(x["soup_done"])) + torch.isclose(x["soup_done"], torch.ones_like(x["soup_done"])))
        assert torch.all((x["soup_cook_time_remaining"] >= -1e-5) * (x["soup_cook_time_remaining"] <= COOK_TIME + 1e-5))
        new_x = []
        for k in NEW_CHANNELS:
            if k in OLD_CHANNELS:
                new_x.append(x[k])
            elif k in CALCULATION:
                new_x.append(CALCULATION[k](x))
            else:
                raise ValueError(f"unknown new feature name k={k}")
        x = torch.stack(new_x, 1)
        x = self.cnn(x)
        
        return x

class CNNBase(nn.Module):
    def __init__(self, args, obs_shape, cnn_layers_params=None):
        super(CNNBase, self).__init__()

        self._use_orthogonal = args.use_orthogonal
        self._activation_id = args.activation_id
        self.hidden_size = args.hidden_size

        self.cnn = CNNLayer(obs_shape, self.hidden_size, self._use_orthogonal, self._activation_id)

    def forward(self, x):
        x = self.cnn(x)
        return x

    @property
    def output_size(self):
        return self.hidden_size
