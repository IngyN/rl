# DeepMind Lab Release Notes

## Next Release

### Minor Improvements:

1.  Improve documentation of how to configure non-hermetic dependencies (Lua,
    Python, NumPy).
2.  Add 'allowHoldOutLevels' setting to allow running of levels that should not
    be trained on, but held out for evaluation.
3.  Add logging library 'common.log', which provides the ability to control which
    log messages are emitted via the setting 'logLevel'.
4.  Update the ioq3 upstream code to the [latest
    state](https://github.com/ioquake/ioq3/tree/29db64070aa0bae49953bddbedbed5e317af48ba).
5.  Lua 5.1 is now downloaded and built from source, and is thus no longer a
    required local dependency.
6.  A minimal version of the "realpath" utility is now bundled with the code,
    and thus "realpath" is no longer a required local dependency.

### Bug Fixes:

1.  Prevent missing sounds from causing clients to disconnect.
2.  Fix a bug in the call of the theme callback 'placeFloorModels', which had
    caused an "'index' is missing" error during compilation of text levels with
    texture sets that use floor models, such as MINESWEEPER, GO, and PACMAN.
3.  Fix bug where levels 'keys_doors_medium', 'keys_doors_random' and
    'rooms_keys_doors_puzzle' would not accept the common 'logLevel' setting.
4.  Expose a 'demofiles' command line flag for the Python random agent, without
    which the agent was not able to record or play back demos.

## release-2018-06-20 June 2018 release

### New Levels:

1.  Psychlab.

    1. contributed/psychlab/glass_pattern_detection
    2. contributed/psychlab/landoltC_identification
    3. contributed/psychlab/motion_discrimination{,_easy}
    4. contributed/psychlab/multiple_object_tracking{,_easy}
    5. contributed/psychlab/odd_one_out

### Bug Fixes:

1.  Let Python level cache set to `None` mean the same as not setting it at all.
2.  Change Python module initialization in Python-3 mode to make PIP packages
    work in Python 3.

### Minor Improvements:

1.  Add support for absl::variant to lua::Push and lua::Read.
2.  The demo `:game` has a new flag `--start_index` to start at an episode index
    other than 0.
3.  Add a console command `dm_pickup` to pick up an item identified by its `id`.
4.  More Python demos and tests now work with Python 3.
5.  Add a shader for rendering decals with transparency.

## release-2018-05-15 May 2018 release

### New Levels:

1.  DMLab-30.

    1.   contributed/dmlab30/psychlab_arbitrary_visuomotor_mapping
    2.   contributed/dmlab30/psychlab_continuous_recognition

2.  Psychlab.

    1.   contributed/psychlab/arbitrary_visuomotor_mapping
    2.   contributed/psychlab/continuous_recognition

### New Features:

1.  Support for level caching for improved performance in the Python module.
2.  Add the ability to spawn pickups dynamically at arbitrary locations.
3.  Add implementations to read datasets including Cifar10 and Stimuli.
4.  Add the ability to specify custom actions via 'customDiscreteActionSpec' and
    'customDiscreteAction' callbacks.

### Bug Fixes:

1.  Fix playerId and otherPlayerId out by one errors in 'game_rewards.lua'.
2.  Require playerId passed to `game:addScore` to be one indexed instead of zero
    indexed and allow `game:addScore` to be used without a playerId.
3.  `game:renderCustomView` now renders the view with top-left as the origin.
    The previous behaviour can be achieved by calling reverse(1) on the returned
    tensor.
4.  Fix a bug in image.scale whereby the offset into the data was erroneously
    ignored.
5.  Fix a typo in a `require` statement in visual_search_factory.lua.
6.  Fix a few erroneous dependencies on Lua dictionary iteration order.
7.  `game:AddScore` now works even on the final frame of an episode.

### Minor Improvements:

1.  Moved .map files into assets/maps/src and .bsp files into assets/maps/built.
    Added further pre-built maps, which removes the need for the expensive
    :map_assets build step.
2.  Allow game to be rendered with top-left as origin instead of bottom-left.
3.  Add 'mixerSeed' setting to change behaviour of all random number generators.
4.  Support for BGR_INTERLEAVED and BGRD_INTERLEAVED observation formats.
5.  Add a Lua API to load PNGs from file contents.
6.  Add 'eyePos' to playerInfo() for a more accurate eye position of player.
    Used in place of player pos + height.
7.  Add support for absl::string_view to lua::Push and lua::Read.
8.  Allow player model to be overridden via 'playerModel' callback.
9.  Add ability to specify custom actions via 'customDiscreteActionSpec' and
    'customDiscreteAction' callbacks.
10. Add `game:console` command to issue Quake 3 console commands directly.
11. Add `clamp` to tensor operations.
12. Add new callback `api:newClientInfo`, allowing each client to intercept
    when players are loading.
13. Skymaze level generation is now restricted to produce only 100000 distinct
    levels. This allows for caching to avoid expensive recompilations.
14. Add cvars 'cg_drawScriptRectanglesAlways' and 'cg_drawScriptTextAlways' to
    enable script rendering when reducedUI or minimalUI is enabled.
15. All pickup types can now choose their movement type separately, and in
    particular, all pickup types can be made static. Two separate table entries
    are now specified for an item, 'typeTag' and 'moveType'.

### Deprecated Features:

1.  Observation format names `RGB_INTERLEAVED` and `RGBD_INTERLEAVED` replace
    `RGB_INTERLACED` and `RGBD_INTERLACED`, respectively. The old format names
    are deprecated and will be removed in a future release.
2.  The pickup item's `tag` member is now called `moveType`. The old name is
    deprecated and will be removed in a future release.

## release-2018-02-07 February 2018 release

### New Levels:

1.  DMLab-30.

     1.  contributed/dmlab30/rooms_collect_good_objects_{test,train}
     2.  contributed/dmlab30/rooms_exploit_deferred_effects_{test,train}
     3.  contributed/dmlab30/rooms_select_nonmatching_object
     4.  contributed/dmlab30/rooms_watermaze
     5.  contributed/dmlab30/rooms_keys_doors_puzzle
     6.  contributed/dmlab30/language_select_described_object
     7.  contributed/dmlab30/language_select_located_object
     8.  contributed/dmlab30/language_execute_random_task
     9.  contributed/dmlab30/language_answer_quantitative_question
    10.  contributed/dmlab30/lasertag_one_opponent_small
    11.  contributed/dmlab30/lasertag_three_opponents_small
    12.  contributed/dmlab30/lasertag_one_opponent_large
    13.  contributed/dmlab30/lasertag_three_opponents_large
    14.  contributed/dmlab30/natlab_fixed_large_map
    15.  contributed/dmlab30/natlab_varying_map_regrowth
    16.  contributed/dmlab30/natlab_varying_map_randomized
    17.  contributed/dmlab30/skymaze_irreversible_path_hard
    18.  contributed/dmlab30/skymaze_irreversible_path_varied
    19.  contributed/dmlab30/psychlab_sequential_comparison
    20.  contributed/dmlab30/psychlab_visual_search
    21.  contributed/dmlab30/explore_object_locations_small
    22.  contributed/dmlab30/explore_object_locations_large
    23.  contributed/dmlab30/explore_obstructed_goals_small
    24.  contributed/dmlab30/explore_obstructed_goals_large
    25.  contributed/dmlab30/explore_goal_locations_small
    26.  contributed/dmlab30/explore_goal_locations_large
    27.  contributed/dmlab30/explore_object_rewards_few
    28.  contributed/dmlab30/explore_object_rewards_many

### New Features:

1.  Basic support for demo recording and playback.

### Minor Improvements:

1.  Add a mechanism to build DeepMind Lab as a PIP package.
2.  Extend basic testing to all levels under game_scripts/levels.
3.  Add settings `minimalUI` and `reducedUI` to avoid rendering parts of the
    HUD.
4.  Add `teleported` flag to `game:playerInfo()` to tell whether a player has
    teleported that frame.
5.  Add Lua functions `countEntities` and `countVariations` to the maze
    generation API to count the number of occurrences of a specific entity or
    variation, respectively.
6.  Add ability to switch/select gadget via actions. Actions are available when
    enabling flags `gadgetSelect` and/or `gadgetSwitch`.

### Bug Fixes:

1.  Fix out-of-bounds access in Lua 'image' library.
2.  Fix off-by-one error in renderergl1 grid mesh rendering.

## release-2018-01-26 January 2018 release

### New Levels:

1.  Psychlab, a platform for implementing classical experimental paradigms from
    cognitive psychology.

    1.  contributed/psychlab/sequential_comparison
    2.  contributed/psychlab/visual_search

### New Features:

1.  Extend functionality of the built-in `tensor` Lua library.
2.  Add built-in `image` Lua library for loading and scaling PNGs.
3.  Add error handling to the env_c_api (version 1.1).
4.  Add ability to create events from Lua scripts.
5.  Add ability to retrieve game entity from Lua scripts.
6.  Add ability create pickup models during level load.
7.  Add ability to update textures from script after the level has loaded.
8.  Add Lua customisable themes. Note: This change renames helpers in
    `maze_generation` to be in lowerCamelCase (e.g. `MazeGeneration` ->
    `mazeGeneration`).
9.  The directory `game_scripts` has moved out of the `assets` directory, and
    level scripts now live separately from the library code in the `levels`
    subdirectory.

### Minor Improvements:

1.  Remove unnecessary dependency of map assets on Lua scripts, preventing
    time-consuming rebuilding of maps when scripts are modified.
2.  Add ability to disable bobbing of reward and goal pickups.
3.  The setting `controls` (with values `internal`, `external`) has been renamed
    to `nativeApp` (with values `true`, `false`, respectively). When set to
    `true`, programs linked against `game_lib_sdl` will use the native SDL input
    devices.
4.  Change LuaSnippetEmitter methods to use table call conventions.
5.  Add config variable for monochromatic lightmaps ('r_monolightmaps'). Enabled
    by default.
6.  Add config variable to limit texture size ('r_textureMaxSize').
7.  api:modifyTexture must now return whether the texture was modified.
8.  Add ability to adjust rewards.
9.  Add ability to raycast between different points on the map.
10. Add ability to test whether a view vector is within an angle range within a
    oriented view frame.

### Bug Fixes:

1.  Increase current score storage from short to long.
2.  Fix ramp jump velocity in level lt_space_bounce_hard.
3.  Fix Lua function 'addScore' from module 'dmlab.system.game' to allow
    negative scores added to a player.
4.  Remove some undefined behaviour in the engine.
5.  Reduce inaccuracies related to angle conversion and normalization.
6.  Behavior of team spawn points now matches that of player spawn points.
    'randomAngleRange' spawnVar must be set to 0 to match previous behavior.

## release-2016-12-06 Initial release
