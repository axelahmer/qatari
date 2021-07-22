A semi-fast Atari Q-Learning program implemented in pytorch and numpy. Built for my own personal research. Use at your own risk.

Install dependencies:

`pip install gym torch tensorboard matplotlib gym[atari]`

Install Atari ROMS: (available from the [Atari 2600 VCS ROM Collection](http://www.atarimania.com/rom_collection_archive_atari_2600_roms.html))

`python -m atari_py.import_roms ./roms/`

Run an example simulation:

`python main.py`

Run a specific game with specific config (other args overwrite the config):

`python main.py --config CONFIG_NAME --game GAME_NAME --seed SEED --qnet MODULE_NAME --double=True/False`

Tensorboard output logged to:

`./results/<game>/<module>_<gameseed>_<timestamp>`

To view results live run tensorboard:

`tensorboard --logdir='./results/<game>'`

Notes:

- Default training configuration conforms to suggestions by Machado et al. 2018.
- Hide/show visualizations by pressing _spacebar_ with game window selected.


OpenAI supported gym games:

'adventure', 'air_raid', 'alien', 'amidar', 'assault', 'asterix', 'asteroids', 'atlantis',
'bank_heist', 'battle_zone', 'beam_rider', 'berzerk', 'bowling', 'boxing', 'breakout', 'carnival',
'centipede', 'chopper_command', 'crazy_climber', 'defender', 'demon_attack', 'double_dunk',
'elevator_action', 'enduro', 'fishing_derby', 'freeway', 'frostbite', 'gopher', 'gravitar',
'hero', 'ice_hockey', 'jamesbond', 'journey_escape', 'kangaroo', 'krull', 'kung_fu_master',
'montezuma_revenge', 'ms_pacman', 'name_this_game', 'phoenix', 'pitfall', 'pong', 'pooyan',
'private_eye', 'qbert', 'riverraid', 'road_runner', 'robotank', 'seaquest', 'skiing',
'solaris', 'space_invaders', 'star_gunner', 'tennis', 'time_pilot', 'tutankham', 'up_n_down',
'venture', 'video_pinball', 'wizard_of_wor', 'yars_revenge', 'zaxxon'
