^^^^^^^^^^^^^^^^^^^^^^^^^^^^^
Changelog for package namosim
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

0.0.1 (2025-08-26)
------------------
* Merge pull request `#12 <https://github.com/Chroma-CITI/namosim/issues/12>`_ from Chroma-CITI/rm_torch
  Update instructions
* docs
* Merge pull request `#11 <https://github.com/Chroma-CITI/namosim/issues/11>`_ from Chroma-CITI/rm_torch
  remove dependency on pytorch
* typo
* fix circular import
* remove imports of ppo agent
* remove imports of ppo agent
* remove imports of ppo agent
* remove dependency on pytorch
* add jacques as maintainer
* ci
* ci
* ci
* skip unit tests for colcon test
* rm pip deps from package.xml
* move pip packages to exec_depend
* ci
* ci
* show colcon test output
* remove dependency on aabbtree
* fix dependencies and colcon test
* add draft-pdf workflow
* docs
* docs
* docs
* docs
* docs
* docs
* docs
* docs
* docs
* docs
* docs
* docs
* docs
* docs
* docs
* docs
* docs
* docs
* docs
* add orcid ids
* docs
* fix citations
* docs
* docs
* docs
* docs and ci
* ci
* docs
* ci
* update github action
* Merge branch 'humble' of github.com:Chroma-CITI/namosim into humble
* update github action
* Merge pull request `#1 <https://github.com/Chroma-CITI/namosim/issues/1>`_ from Chroma-CITI/paper
  Paper
* update github action
* add github action
* add github action
* paper
* paper
* paper
* paper
* paper
* paper
* paper
* paper
* paper
* paper
* paper
* paper
* paper
* Merge branch 'joss' into 'humble'
  add joss paper
  See merge request chroma/namo/namosim!196
* add joss paper
* Merge branch 'three_robots' into 'humble'
  Three robots
  See merge request chroma/namo/namosim!194
* use absolute cost values when compuing evasion cost
* is_for_deadlock
* fix tests
* format
* types
* fix types
* clean up
* clean up
* clean up
* simplify conflict detection
* simplifying conflict detection
* clean up
* clean up
* remove unnecessary aabb trees
* simplifying conflict detection
* Merge branch 'stilman_rrt' into 'humble'
  stilman rrt
  See merge request chroma/namo/namosim!191
* PoseModel -> Pose2D
* use named tuple for PoseModel
* fix set_current_action_index
* fix set_current_action_index
* add function to set current action index on navigation plan
* fix tests
* fix stealing movable test
* fix tests
* test kd tree
* use obstacle pose not robot pose for social cost
* penalize rotations
* stilman rrt
* add collision margin for robot_obstacle_polyon in RRT* agent
* add minim discretization of poses in RRT tree
* cache poses for rrt star
* collision bug fix
* fair comparison
* stilman rrt
* default to diff drive
* stilman rrt
* stilman rrt
* Merge branch 'humble' into stilman_rrt
* Merge branch 'triangles' into 'humble'
  fix polygon triangulation bug
  See merge request chroma/namo/namosim!192
* propagate non-zero exit codes in tests
* fix polygon triangulation bug
* use correct center or rotation in rrt
* stilman rrt
* Merge branch 'minor' into 'humble'
  minor updates for stolen-obstacle conflicts
  See merge request chroma/namo/namosim!190
* minor updates for stolen-obstacle conflicts
* Merge branch 'get_movables' into 'humble'
  change return type of get_movable_obstacles to a dict
  See merge request chroma/namo/namosim!189
* change return typ of get_movable_obstacles to a dict
* Merge branch 'goal_polygon' into 'humble'
  add kd tree
  See merge request chroma/namo/namosim!188
* kd tree
* add kd tree
* Merge branch 'goal_polygon' into 'humble'
  use goal polygon from svg instead of point buffer
  See merge request chroma/namo/namosim!187
* fix tests
* use goal polygon from svg instead of point buffer
* fix goal shape
* Merge branch 'rm_namoros' into 'humble'
  move namoros out into a separate repo
  See merge request chroma/namo/namosim!186
* update docs
* move namoros out into a separate repo
* Merge branch 'obstacle' into 'humble'
  automatically add spawn obstacles from scenario svg
  See merge request chroma/namo/namosim!185
* bug fix
* Merge branch 'obstacle' of gitlab.inria.fr:chroma/namo/namosim into obstacle
* non-circular rrt
* readme
* progress on stolen obstacle conflicts
* stolen obstacle conflict
* rm aruco markers submodule
* tuning
* params
* automatically add spawn obstacles from scenario svg
* add config param to automatically add movable obstacles to the map
* Merge branch 'db' into 'humble'
  fix docs and show manip search in rviz
  See merge request chroma/namo/namosim!184
* fix docs and show manip search in rviz
* Merge branch 'dbrown-humble-patch-bd78' into 'humble'
  Update file README.md
  See merge request chroma/namo/namosim!183
* Update file README.md
* Merge branch 'docs' into 'humble'
  docs
  See merge request chroma/namo/namosim!182
* docs
* fix ign service command
* ci
* docs
* Merge branch 'gz' into 'dev'
  move namo gz plugin into its own package
  See merge request chroma/namo/namosim!180
* move namo gz plugin into its own package
* ci
* docs
* docs
* Merge branch 'ros' into 'dev'
  add devcontainer
  See merge request chroma/namo/namosim!179
* ci
* add dev container
* ci
* ci
* ci
* Merge branch 'ros' into 'dev'
  combine namoros and namosim
  See merge request chroma/namo/namosim!178
* add dockerignore
* fix dep
* update submodule url
* add dockerfile
* update submodule url
* ci
* docs
* docs
* ci
* ci
* cleanup
* set pip package versions
* types
* combine namoros and namosim
* fix pytest version
* Edit LICENSE
* Merge branch 'ros' into 'dev'
  make namosim a ros package
  See merge request chroma/namo/namosim!177
* fix concave hull
* use unary_union
* cleanup
* type checking
* docs
* add numpy stl
* add triangle to requirements
* minor bug
* use triangle instead of earcut
* ci
* types
* types
* ci
* ros
* Merge branch 'rrt' into 'dev'
  rrt
  See merge request chroma/namo/namosim!175
* fix tests
* docs
* docs
* docs
* docs
* docs
* docs
* docs
* rrt
* Merge branch 'dev' into rrt
* add option to not draw grid lines
* rrt
* Merge branch 'conflicts' into 'dev'
  draw grid lines
  See merge request chroma/namo/namosim!174
* draw grid lines
* Merge branch 'conflicts' into 'dev'
  Conflicts
  See merge request chroma/namo/namosim!173
* type check
* ignore conflicts that reoccur while evading
* Merge branch 'conflicts' into 'dev'
  test space conflict
  See merge request chroma/namo/namosim!172
* fix tests
* minor
* test space conflict
* fix type check
* fix global install
* fix install in editable mode
* docs
* Merge branch 'docs' into 'dev'
  docs
  See merge request chroma/namo/namosim!171
* docs
* Merge branch 'postpone' into 'dev'
  refactoring postpones
  See merge request chroma/namo/namosim!169
* refactoring postpones
* Merge branch 'grab_dist' into 'dev'
  add config params for grab start and end distances
  See merge request chroma/namo/namosim!168
* add config params for grab start and end distances
* Merge branch 'multi_robot' into 'dev'
  updates to support synchronization plan with observed state
  See merge request chroma/namo/namosim!167
* fix tests
* updates to support synchronization plan with observed state
* publish namespace text marker
* fix ci
* support passing a callback group to the ros publisher
* Merge branch 'rviz' into 'dev'
  cleanup rviz visualization
  See merge request chroma/namo/namosim!165
* cleanup rviz visualization
* minor param adjustments
* Merge branch 'svg2stl' into 'dev'
  update svg2stl script to use wall geometries from svg instead of the occupancy...
  See merge request chroma/namo/namosim!163
* fix ci
* cleanup
* update svg2stl script to use wall geometries from svg instead of the occupancy grid, to generate meshes
* Merge branch 'opencv_headless' into 'dev'
  use opencv-headless
  See merge request chroma/namo/namosim!161
* use opencv-headless
* Merge branch 'namosim-private-dev' into 'dev'
  merge namosim private and fix conflicts
  See merge request chroma/namo/namosim!160
* merge namosim private and fix conflicts
* Merge branch 'dev' into 'dev'
  Refactor/simplify the svg scenario format
  See merge request chroma/namo/namosim!157
* Refactor/simplify the svg scenario format
* Merge branch 'dbrown-dev-patch-47654' into 'dev'
  Update LICENSE
  See merge request chroma/namo/namosim!155
* Update LICENSE
* Merge branch 'cleanup' into 'dev'
  remove unused params
  See merge request chroma/namo/namosim!152
* remove unused params
* Merge branch 'cleanup' into 'dev'
  remove unused integration tests and unused parameters
  See merge request chroma/namo/namosim!150
* remove unused integration tests and unused parameters
* Merge branch 'minor_udpates' into 'dev'
  minor updates for video examples
  See merge request chroma/namo/namosim!149
* minor updates for video examples
* Merge branch 'minor_udpates' into 'dev'
  minor updates
  See merge request chroma/namo/namosim!148
* minor updates
* Merge branch 'combined_cost' into 'dev'
  minor bug fix related to combined cost
  See merge request chroma/namo/namosim!147
* add scenarios
* minor bug fix related to combined cost
* minor bug fix related to combined cost
* Merge branch 'paper_scenarios' into 'dev'
  add example scenarios for iros paper
  See merge request chroma/namo/namosim!146
* clean up
* add example scenarios for iros paper
* Merge branch 'notebook' into 'dev'
  update notebook to fix makespan
  See merge request chroma/namo/namosim!143
* update notebook to fix makespan
* Merge branch 'snamo_distance_dr' into 'dev'
  add parameter to test snamo with distance-based deadlock resolution
  See merge request chroma/namo/namosim!141
* update notebook
* update data models
* minor updates
* update notebook
* minor updates
* update scripts
* update notebook
* add parameter to test snamo with distance-based deadlock resolution
* Merge branch 'notebook' into 'dev'
  update legend
  See merge request chroma/namo/namosim!140
* update legend
* Merge branch 'analysis' into 'dev'
  update notebook
  See merge request chroma/namo/namosim!139
* update notebook
* Merge branch 'ros' into 'dev'
  update notebook
  See merge request chroma/namo/namosim!138
* update notebook
* Merge branch 'ros' into 'dev'
  make ros optional
  See merge request chroma/namo/namosim!136
* update notebook
* update notebook
* update notebook
* add scripts for wg multi shape
* fix type errors
* fix lint error
* make ros optional
* update willow
* Merge branch 'wg' into 'dev'
  add script to generate willow garage multi-shape scenarios
  See merge request chroma/namo/namosim!135
* add script to generate willow garage multi-shape scenarios
* Merge branch 'analysis' into 'dev'
  update v2 notebook
  See merge request chroma/namo/namosim!134
* update v2 notebook
* Merge branch 'wg_multi_shape' into 'dev'
  add willow garage multi-shape scenario
  See merge request chroma/namo/namosim!133
* Merge branch 'dev' into wg_multi_shape
* Merge branch 'results_by_objective' into 'dev'
  compute results per goal instead of per sim
  See merge request chroma/namo/namosim!132
* add willow garage multi-shape scenario
* add new notebook
* cleanup
* cleanup
* fix errors
* cleanup
* compute results per goal instead of per sim
* Merge branch 'bound' into 'dev'
  reduce solution_interval_bound_percentage
  See merge request chroma/namo/namosim!131
* reduce solution_interval_bound_percentage
* Merge branch 'conf' into 'dev'
  fix minor conflict check issue
  See merge request chroma/namo/namosim!130
* fix minor conflict check issue
* Merge branch 'gen' into 'dev'
  bump solution interval bound percentage
  See merge request chroma/namo/namosim!129
* bump solution interval bound percentage
* Merge branch 'gen' into 'dev'
  update exp scripts
  See merge request chroma/namo/namosim!128
* update exp scripts
* Merge branch 'timeout' into 'dev'
  increase planning timeout
  See merge request chroma/namo/namosim!127
* increase planning timeout
* Merge branch 'willow' into 'dev'
  add script to launch willow garage experiments
  See merge request chroma/namo/namosim!126
* add script to launch willow garage experiments
* Merge branch 'citi' into 'dev'
  add script to launch citi experiments
  See merge request chroma/namo/namosim!125
* update base citi-lab scenario
* add script to launch citi experiments
* Merge branch 'astar_evasion' into 'dev'
  use a-star search for social evasion
  See merge request chroma/namo/namosim!123
* Merge branch 'plots' into 'dev'
  update jupyter notebook
  See merge request chroma/namo/namosim!122
* use a-star search for social evasion
* update jupyter notebook
* Merge branch 'plots' into 'dev'
  display std on plots
  See merge request chroma/namo/namosim!121
* display std on plots
* Merge branch 'collisions' into 'dev'
  simplify arc bounding box function
  See merge request chroma/namo/namosim!117
* revert timeout change
* remove unused bb_type arg
* minor
* fix error in collision detection during conflict detection
* Merge branch 'db/report' into 'dev'
  generate csv and add data analysis notebook
  See merge request chroma/namo/namosim!119
* generate csv and add notebook
* Merge branch 'db/report' into 'dev'
  minor update to report
  See merge request chroma/namo/namosim!118
* minor update to report
* cleanup
* cleanup
* simplify arc bounding box function
* Merge branch 'willow' into 'dev'
  add script to generate willow garage scenarios
  See merge request chroma/namo/namosim!116
* add script to generate willow garage scenarios
* Merge branch 'conflicts' into 'dev'
  add conflicts to report
  See merge request chroma/namo/namosim!115
* fix pipeline errors
* fix errors
* add num steps to sim report
* add conflicts to report
* Merge branch 'db/results' into 'dev'
  graph additional metrics
  See merge request chroma/namo/namosim!113
* graph additional metrics
* Merge branch 'db/timetous' into 'dev'
  reset agent after planning timeout
  See merge request chroma/namo/namosim!112
* reset agent after planning timeout
* Merge branch 'naive-evasion' into 'dev'
  minor updates to simulation report
  See merge request chroma/namo/namosim!109
* minor report updates
* minor bug fix in report
* Merge branch 'naive-evasion' into 'dev'
  fix get_min_dist_to_others
  See merge request chroma/namo/namosim!108
* get_min_dist_to_others
* Merge branch 'naive-evasion' into 'dev'
  handle planning timeouts and add them to report
  See merge request chroma/namo/namosim!107
* minor bug fix
* cleanup
* cleanup
* cleanup
* handle planning timeouts and add them to report
* Merge branch 'naive-evasion' into 'dev'
  updates for nonsocial evasion and related experiments
  See merge request chroma/namo/namosim!106
* fix compare-results script
* fix lint error
* updates for nonsocial evasion and related experiments
* Merge branch 'report' into 'dev'
  bump python version for 3.10
  See merge request chroma/namo/namosim!105
* update poetry lock
* bump python version
* Merge branch 'report' into 'dev'
  use A* for non-social evasion
  See merge request chroma/namo/namosim!104
* use A* for non-social evasion
* Merge branch 'report' into 'dev'
  fix simulation report json serialization
  See merge request chroma/namo/namosim!103
* fix simulation report json serialization
* Merge branch 'willow' into 'dev'
  add willow garage scenario
  See merge request chroma/namo/namosim!102
* revert parameter default
* add more obstacles
* add willow garage scenario
* Merge branch 'deadlocks' into 'dev'
  update launch experiments script
  See merge request chroma/namo/namosim!101
* update launch experiments script
* Merge branch 'namo-deadlock' into 'dev'
  fix scenario generation
  See merge request chroma/namo/namosim!100
* fix scenario generation
* Merge branch 'namo-deadlock' into 'dev'
  add non-social evasion strategy
  See merge request chroma/namo/namosim!99
* add non-social evasion strategy
* Merge branch 'namo-deadlock' into 'dev'
  encapsulate deadlock resolution logic in a function
  See merge request chroma/namo/namosim!98
* encapsulate deadlock resolution logic in a function
* Merge branch 'namo-deadlock' into 'dev'
  cleanup superfless else before return
  See merge request chroma/namo/namosim!97
* cleanup superfless else before return
* Merge branch 'namo-deadlock' into 'dev'
  use exclude list for pyright
  See merge request chroma/namo/namosim!96
* use exclude list for pyright
* Merge branch 'db/translate' into 'dev'
  refactor actions
  See merge request chroma/namo/namosim!95
* refactor actions
* Merge branch 'db/copy' into 'dev'
  be careful with deepcopy
  See merge request chroma/namo/namosim!94
* be careful with deepcopy
* Merge branch 'db/experiment' into 'dev'
  use light_copy for agent world copy
  See merge request chroma/namo/namosim!92
* cleanup agent copy function and add more types to stillman agent
* use light_copy for agent world copy
* properly use resolve_conflicts param
* reduce timeout
* add snamo_ncr variant to experiments
* Merge branch 'db/experiment' into 'dev'
  add adhoc python script to visualize results
  See merge request chroma/namo/namosim!91
* bug fix
* add adhoc python script to visualize results
* add adhoc python script to visualize results
* Merge branch 'db/experiment' into 'dev'
  simplify report generation
  See merge request chroma/namo/namosim!90
* simplify report generation
* Merge branch 'db/experiment' into 'dev'
  update scripts for launching experiments
  See merge request chroma/namo/namosim!89
* cleanup
* add parameters to enable/disable conflict resolution and deadlock resolution
* update scripts for launching experiments
* Merge branch 'db/teleop' into 'dev'
  teleop agent
  See merge request chroma/namo/namosim!88
* Merge branch 'db/conflicts' into 'dev'
  handle robot-robot space conflicts that occur from a grab
  See merge request chroma/namo/namosim!87
* handle keydown and keyup
* add grab/release actions to teleop agent
* fix bug
* add teleop agent
* ignore rotations less than 1e-6
* handle robot-robot space conflicts that occurr from a grab
* minor update to report visualization
* Merge branch 'db/conflicts' into 'dev'
  show total goals in report graph
  See merge request chroma/namo/namosim!86
* fix conflicts with main
* show total goals in report
* Merge branch 'db/conflicts' into 'dev'
  improvements for robot conflict detection
  See merge request chroma/namo/namosim!84
* improvements for robot conflict detection
  * Updates conflict detection function to handle an edge case where there are two transfer paths back-to-back.
  * Updates conflict detection function to handle the case where a collision is detected with an obstacle the robot is currently holding which causes the robot to be in conflict with itself.
  * Minor refactoring in the conflict detection functions to simplify the code and make it more readable.
* minor bug fix
* properly handle case where there are two transfer paths back-to-back
* reduce check horizon to 10 steps
* improvements for robot conflict detection
* Merge branch 'db/reports' into 'dev'
  enhance report visualization
  See merge request chroma/namo/namosim!83
* enhance report visualization
* Merge branch 'dev' into 'main'
  prepare for release v0.0.7
  See merge request chroma/namo/namosim!82
* prepare for release v0.0.7
* Merge branch 'db/conflicts' into 'dev'
  tighten robot conflict radius and reduce check horizon
  See merge request chroma/namo/namosim!81
* tighten robot conflict radius and reduce check horizon
  Also:
  * make sure robots start far enough apart in generated scenarios
  * don't use robot min inflation grid during manipulation search because it could cause the robot to replan while inside static obstacle grid cell
  * fix bug where compute_evasion() inadvertently re-activates the main robot in the robot-inflated grid
* Merge branch 'db/conflicts' into 'dev'
  make sure release distance is larger than cell size
  See merge request chroma/namo/namosim!80
* make sure release distance is larger than cell size
* Merge branch 'db/conflicts' into 'dev'
  handle case where conflicting agent is on the current agent's goal
  See merge request chroma/namo/namosim!79
* Merge branch 'db/conflicts' into 'dev'
  update avoid list before recursive call
  See merge request chroma/namo/namosim!78
* Merge branch 'db/conflicts' into 'dev'
  debugging conflict resolution
  See merge request chroma/namo/namosim!77
* fix tests
* handle case where conflicting agent is on the current agents goal
* fix avoid list
* fix avoid list
* fix avoid list
* update avoid list before recursive call
* add comments and cleanup
* omit empty rotations
* fix conflict radius
* add function to compute robot conflict radius
* cleanup
* cleanup
* fix tests
* fix issue in binary grid map boundary calculation
* cleanup
* raise exceptions for cases that should never happen
* debugging conflict resolution
* Merge branch 'dev' into 'main'
  Prepare for release v0.0.6
  See merge request chroma/namo/namosim!76
* Merge branch 'db/scenario-gen' into 'dev'
  fix issue in scenario generation
  See merge request chroma/namo/namosim!75
* start running experiments
* Merge branch 'db/scenario-gen' into 'dev'
  updates for generated scenarios
  See merge request chroma/namo/namosim!74
* minor
* fix issue in scenario generation
* fix signed angle bug
* add 1-robot scenario
* working on generated scenarios
* Merge branch 'db/scenario-gen' into 'dev'
  start migrating scenario generation
  See merge request chroma/namo/namosim!73
* Merge branch 'db/reports' into 'dev'
  progress on report generation
  See merge request chroma/namo/namosim!72
* Merge branch 'experimental-unify-agent' into 'dev'
  Experimental unify agent
  See merge request chroma/namo/namosim!68
* remove unnecessary copy
* add script to graph results
* cleanup visualization markers
* add experiment launch script
* add script to generate citi-lab scenarios
* write generate scenarios to a specified output dir
* add pause functionality
* add 4-robot experiment and unit test for obstacle-on-goal
* fix tests
* add comments
* progress on scenario generation
* fix step count
* remove taboo zones (not used)
* start migrating scenario generation
* fix lint error
* clean up
* add ability to compare two reports
* plot report
* simplify logs dir
* use pydantic for report data model
* progress on report generation
* add experiment for intersections scenario without social cost
* progress on report generation
* merge dev
* Merge branch 'dev' into 'main'
  Prepare for release v0.0.5
  See merge request chroma/namo/namosim!71
* Merge branch 'db/bug-fix' into 'dev'
  update docs
  See merge request chroma/namo/namosim!70
* update docs
* Merge branch 'db/bug-fix' into 'dev'
  bug fix for TransitPath.from_poses()
  See merge request chroma/namo/namosim!69
* clean up
* fix tests
* clean up
* bug fix for TransitPath.from_poses()
* rename behaviors to agents
* finish unifying robot and behavior
* init agents
* init agents
* set agent worlds
* progress on unifying robot and behavior
* progress unifying robot and behavior
* Merge branch 'db/world-v2' into 'dev'
  more types/type-checking
  See merge request chroma/namo/namosim!67
* progress on unifying robot and behavior
* fix lint error
* minor visualization improvement
* Merge branch 'db/world-v2' into 'dev'
  bug fix for stolen movable conflict detection
  See merge request chroma/namo/namosim!66
* fix stolen movable conflict detection
* add stolen obstacle conflict test
* add 1-robot-2-obstacles social test
* Merge branch 'db/world-v2' into 'dev'
  pass robot uid to plan
  See merge request chroma/namo/namosim!65
* pass robot uid to plan
* Merge branch 'db/world-v2' into 'dev'
  ignore collisions detected during act step
  See merge request chroma/namo/namosim!64
* fix checks
* ignore collisions detected during act step
* Merge branch 'db/world-v2' into 'dev'
  unify world scenario files in a single svg
  See merge request chroma/namo/namosim!57
* fix tests
* add citi lab scenario
* fix 2-robot intersections scenario
* use 30 degree rotation angle
* fix tests
* use rotation unit angle param
* remove v2s
* extract map bounds from svg viewbox
* fix tests
* update rviz config
* clean up
* scale all markers relative to the robot radius
* work on migrating after-the-feast
* cleanup
* Merge branch 'db/world-v2-temp' into 'db/world-v2'
  intermediate progress on world v2
  See merge request chroma/namo/namosim!63
* adapt two-rooms scenario
* Merge branch 'dev' into 'main'
  prepare for release v0.0.4
  See merge request chroma/namo/namosim!61
* cleanup
* progress on world v2
* Merge branch 'dev' into db/world-v2
* Merge branch 'db/path' into 'dev'
  fix path rendering in rviz
  See merge request chroma/namo/namosim!62
* fix tests
* fix tests
* fix path rendering in rviz
* leave version num in pyproject.toml at 0.0.0 as this will be determined by release tags
* Merge branch 'db/debug-evasion' into 'dev'
  debug evasion plan
  See merge request chroma/namo/namosim!60
* clean up
* clean up
* resolve bug
* debug evasion subroutine
* Merge branch 'db/docs' into 'dev'
  minor bug fix
  See merge request chroma/namo/namosim!59
* minor bug fix
* Merge branch 'db/docs' into 'dev'
  add scenario for moving two obstacles
  See merge request chroma/namo/namosim!58
* add scenario files
* update comments
* cleanup nav-only test
* add scenario for moving two obstacles
* add comments
* fix tests
* fix conflicts
* Merge branch 'db/stilman' into 'dev'
  implement stilman-only
  See merge request chroma/namo/namosim!56
* minor bug fix
* remove scaling from stilman-only test
* get unit test working for stilman-only
* unify world scenario files in a single svg
* progress on stilman only
* progress on stilman only
* progess on vanila stilman behavior
* Merge branch 'db/new-scenario' into 'dev'
  add another multi-robot scenario
  See merge request chroma/namo/namosim!55
* add another multi-robot scenario
* Merge branch 'db/navigation-only' into 'dev'
  navigation only behavior
  See merge request chroma/namo/namosim!54
* add unit test for navigation only behavior
* Merge branch 'db/refactoring' into 'dev'
  minor refactoring
  See merge request chroma/namo/namosim!53
* implement navigation only behavior
* minor refactoring
* Merge branch 'db/separate-concerns' into 'dev'
  pass ros publisher as a parameter to the behavior
  See merge request chroma/namo/namosim!52
* cleanup
* improve handling of cleaning up conflict checks
* cleanup
* pass ros publisher to behavior as a param, not an instance arg
* remove rviz publish from behavior sensing step
* Merge branch 'db/orientation' into 'dev'
  fix a bug in converting the robot orientation mark to svg
  See merge request chroma/namo/namosim!50
* fix lint errors
* remove parallel think
* rebase onto dev
* Merge branch 'db/params' into 'dev'
  remove unused/deprecated discretization data params
  See merge request chroma/namo/namosim!48
* Merge branch 'db/svg' into 'dev'
  render world in tk window at each step of simulation
  See merge request chroma/namo/namosim!47
* add 3-robot experiment
* add support for robots thinking in parallel
* add comments
* remove unused/deprecated discretization data params
* properly handdle deactivate_gui variable
* tidy up display window rendering
* experimenting with display window
* Merge branch 'db/structure' into 'dev'
  flatten package structure
  See merge request chroma/namo/namosim!46
* flatten package structure
* Merge branch 'doc/scenario' into 'dev'
  improve documentation for creating a scenario
  See merge request chroma/namo/namosim!45
* improve documation for creating a scenario
* Merge branch 'doc/scenario' into 'dev'
  add minimal documentation for creating a scenario
  See merge request chroma/namo/namosim!44
* Merge branch 'dev' into doc/scenario
* Merge branch 'db/types' into 'dev'
  more typings
  See merge request chroma/namo/namosim!43
* add minimal documentation for creating a scenario
* Merge branch 'db/types' into 'dev'
  add more types to ros publisher
  See merge request chroma/namo/namosim!42
* Merge branch 'db/path-rendering' into 'dev'
  render path as a triangulated polygon
  See merge request chroma/namo/namosim!41
* more types
* more types
* more types in ros publisher
* more types in ros publisher
* add more types to ros publisher
* add unit tests and docstrings
* fix lint error
* render path as a triangulated polygon
* Merge branch 'dev' into 'main'
  Merge dev into main
  See merge request chroma/namo/namosim!40
* Merge branch 'db/typings' into 'dev'
  add types to binary occupancy grid
  See merge request chroma/namo/namosim!39
* add types to binary occupancy grid
* Merge branch 'db/typings' into 'dev'
  add more type checking
  See merge request chroma/namo/namosim!38
* add more type checking
* Merge branch 'db/typings' into 'dev'
  skip ci builds for release tags
  See merge request chroma/namo/namosim!37
* remove semantic release because it is not designed for gitflow
* remove semantic release because it is not designed for gitflow
* 0.0.1
  Automatically generated by python-semantic-release
* Merge branch 'dev' into 'main'
  Dev
  See merge request chroma/namo/namosim!36
* skip ci builds for release tags
* Merge branch 'db/typings' into 'dev'
  fix: add more typings to behavior modules and fix rviz bug
  See merge request chroma/namo/namosim!35
* fix: add more typings to behavior modules and fix rviz bug
* fix: add more typings to behavior modules and fix rviz bug
* 0.0.0
  Automatically generated by python-semantic-release
* Merge branch 'db/releases' into 'main'
  fix automatic release
  See merge request chroma/namo/namosim!34
* fix automatic release
* Merge branch 'db/releases' into 'main'
  setup automatic releases
  See merge request chroma/namo/namosim!33
* setup automatic releases
* Merge branch 'db/releases' into 'main'
  setup automatic releases
  See merge request chroma/namo/namosim!32
* setup automatic releases
* Merge branch 'db/releases' into 'main'
  setup automatic releases
  See merge request chroma/namo/namosim!31
* setup automatic releases
* Merge branch 'db/base-plan' into 'main'
  break cyclic dependency between Behavior and Path classes
  See merge request chroma/namo/namosim!30
* Merge branch 'db/base-plan' into 'main'
  Improve usage of Plan classes
  See merge request chroma/namo/namosim!29
* break cyclic dependency between Behavior and Path classes
* more type checks and light refactoring to avoid cyclic imports
* Merge branch 'db/multi-robot-test' into 'main'
  add unit test for basic multi-robot scenario
  See merge request chroma/namo/namosim!28
* fix type checks
* remove ros publisher singleton to fix unit tests
* bump ci runner size
* add unit test for basic multi-robot scenario
* Merge branch 'db/docs' into 'main'
  minor docs update
  See merge request chroma/namo/namosim!27
* minor docs update
* Merge branch 'db/docs' into 'main'
  add more content to docs
  See merge request chroma/namo/namosim!26
* typo
* typo
* add more content to docs
* add more content to docs
* Merge branch 'db/docs' into 'main'
  add link to docs site in readme
  See merge request chroma/namo/namosim!25
* add link to docs site in readme
* Merge branch 'db/docs' into 'main'
  fix doc page publish
  See merge request chroma/namo/namosim!24
* fix doc page publish
* Merge branch 'db/docs' into 'main'
  Db/docs
  See merge request chroma/namo/namosim!23
* generate gitlab docs page
* generate gitlab docs page
* Merge branch 'db/docs' into 'main'
  generate gitlab docs page
  See merge request chroma/namo/namosim!22
* generate gitlab docs page
* Merge branch 'db/docs' into 'main'
  add readthedocs
  See merge request chroma/namo/namosim!21
* add readthedocs boilerplate
* Merge branch 'db/readme' into 'main'
  add screenshot to README
  See merge request chroma/namo/namosim!20
* add screenshot to README
* Merge branch 'db/sim-model' into 'main'
  progress on sim config file data model
  See merge request chroma/namo/namosim!19
* fix checks
* use simulation config data model
* progress on sim config file data model
* Merge branch 'db/custom-scenario' into 'main'
  add custom scenario in unit tests
  See merge request chroma/namo/namosim!18
* Merge branch 'db/drop-ros1' into 'main'
  more cleanup
  See merge request chroma/namo/namosim!17
* add custom scenario
* fix type checks
* fix lint errros
* more cleanup
* Merge branch 'db/drop-ros1' into 'main'
  more type checking
  See merge request chroma/namo/namosim!16
* Merge branch 'db/drop-ros1' into 'main'
  drop support for ros1 to simplify ros publisher
  See merge request chroma/namo/namosim!15
* more type checking
* drop ros1 support
* remove unused import
* drop support for ros1 for simplicity
* add some minimal type checking
* Merge branch 'db/models' into 'main'
  refactoring in world and simulator files
  See merge request chroma/namo/namosim!14
* refactoring in world and simulator files
* Merge branch 'db/models' into 'main'
  progress on pydantic data models for world objects
  See merge request chroma/namo/namosim!13
* minor cleanup
* minor
* Merge branch 'db/bug' into 'main'
  more cleanup
  See merge request chroma/namo/namosim!12
* bug fix
* remove unused arg
* progess on world data models
* Merge branch 'db/bug' into 'main'
  bugfix
  See merge request chroma/namo/namosim!11
* more cleanup
* start adding type hints
* start adding type hints
* Merge branch 'db/merge' into 'main'
  merge changes from s-namo-sim-private
  See merge request chroma/namo/namosim!10
* cleanup
* bugfix
* merge changes from s-namo-sim-private
* Merge branch 'db/min-scenario' into 'main'
  add minimal scenario to unit tests
  See merge request chroma/namo/namosim!9
* try ros-iron-desktop-full
* change ci image
* change ci image
* change ci image
* add minimal scenario to unit tests
* Merge branch 'db/check' into 'main'
  remove box2d and other cleanup
  See merge request chroma/namo/namosim!8
* remove box2d and other cleanup
* Merge branch 'db/ci' into 'main'
  add runner tags
  See merge request chroma/namo/namosim!7
* test
* add swig
* add swig
* add poetry to path
* change ci image
* add runner tags
* Merge branch 'db/tests' into 'main'
  add ci file
  See merge request chroma/namo/namosim!6
* add ci file
* Merge branch 'db/tests' into 'main'
  more cleanup
  See merge request chroma/namo/namosim!4
* Merge branch 'db/readme' into 'main'
  add submodule for iros 2021 data
  See merge request chroma/namo/namosim!3
* Merge branch 'db/readme' into 'main'
  remove dead code and fix lint errors
  See merge request chroma/namo/namosim!2
* fix script
* move tests out of main package
* add submodule for iros 2021 data
* add lint and format tasks
* add .vscode folder for shared settings
* remove dead code
* Merge branch 'db/readme' into 'main'
  update readme
  See merge request chroma/namo/namosim!1
* update readme
* cleanup
* more lint changes
* Merge branch 'db/pyproject' into db/lint
* fix box2d git url
* update readme
* remove pre-commit, for now
* lint the code with ruff and add pre-commit hook
* minor readme update
* minor readme update
* replace setup.py with pyproject.toml
* Continue refactor of RosPublisher: plan
* Remove useless svg_test.py file
* Continue refactor of RosPublisher: q_manip_for_obs
* Continue refactor of RosPublisher: GoalObserver
* Remove all other deprecated Ros conversions and publish functions from RosPublisher
* Remove deprecated a_star and multi_goal_a_star publish functions from RosPublisher
* Remove deprecated a_star and multi_goal_a_star publish functions from RosPublisher
* Remove deprecated path_grid_cells publish functions from RosPublisher
* Remove deprecated min_max_inflated publish functions from RosPublisher
* Merge branch 'master' of https://gitlab.inria.fr/brenault/s-namo-sim-private
* Continue refactor of RosPublisher
* Update README.md
* Fix some regressions and bugs linked to newer libraries versions
* Merge branch 'master' of https://gitlab.inria.fr/brenault/s-namo-sim-private
* Add missing dep in requirements
* Merge branch 'dwb' into 'master'
  add example to readme
  See merge request brenault/s-namo-sim-private!5
* add example to readme
* Fix deprecation warnings
* Continue RosPublisher refactor
* Continue RosPublisher refactor: user default rate parameter
* Continue RosPublisher refactor: stop using hardcoded entity colors - working.
* Continue RosPublisher refactor: stop using hardcoded entity colors.
* Continue refactor of ros publisher
* Slightly adapt RosPublisher config
* Add 6-digit hex color conversion to floats
* Comment Box2D requirement for future complete removal
* Adapt to latest version of Shapely
* Big update, add lots of fixes and mainly also add ROS2 compatibility.
* Uncomment NAMO scenario for 3 rooms - 3 robots scenario
* Continue fixing stat registration
* Fix exchange path computation in evasion computation method
* Start update of statistics criteria
* Better define t+1 inflation radius
* Don't detect conflicts for wait steps
* Make sure only RobotRobot conflicts are considered for potential deadlock detection.
* Don't use forbidden_evasion_cells set for other robot evasion computation, it does not make sense.
* When replanning, consider that obstacles in ConcurrentGrab conflicts are being held by the other robot that is part of the conflict.
* Compute evasion with n-1 strategy and updated wait time at evasion configuration.
* Fix plan counting
* Fix merge
* Regularisation commit after merge
* Merge branch 'NoRecoveryPath' into 'master'
  Remove Recovery Path "Hack".
  See merge request brenault/s-namo-sim-private!3
* Update conflict definitions and detection, mainly so that t+1 predictions work reliably.
* Fix implementation of polygon removal from grid
* Correctly update entity_to_agent attribute when entities are ignored in light copy
* Fix inscribed radius computation
* Update README.md to fit with public version.
* Use shapely functions to get inscribed and circumscribed circles radiuses (smaller, faster, more accurate)
* Remove RecoveryPath
* Merge branch 'clean_b2sim_out' into 'master'
  Great update of master without b2sim
  See merge request brenault/s-namo-sim-private!2
* Clean all traces of Box2D version of code.
* Better implementation of A* and Dijkstra, with dynamic goal. All that is left to do is to add a update_graph function that will properly invalidate parts of the search tree and get it back to a state where the search can be restarted.
* Update graph search test.
* Rewrite local coordination stratedy with evasion capability, and fix local opening detection implementation.
* Add verification of no-collision at predicted t+1 configuration of robots
* Update conflict definition
* Use atan2 for direction vector, add extra checks, and implement tentative Circle collision shape.
* Change nb of plan computations counting method.
* Auto-add newline at EOF
* Finish big dataset structure update and add mr-namo tests
* Add extra check for edge case of dijkstra grid search.
* Update rviz vizualization
* Use 8-n propagation for social cost model wave
* Change binary occupancy grid update logic for deleted entities.
* Fix tmux launcher
* Dataset structure upate
* First batch of fixes after ICRA2022 Submission
* Update stats generation for ICRA paper
* Fix recurrent  exception
* Last minute commit for expes
* Last minute commit for expes
* Last minute commit for expes
* Last minute commit for expes
* Make entity generation possible in world export (still need to fix world update properly).
* Remove deprecated issues savefile.
* Rewrite stats analysis to use multiprocessing to generate stats within reasonable amount of time.
* Fix scenario generation so that no robots may overlap at init pose.
* Prepare for 4 robots after the feast experiment
* Fix direction generation in scenario generation.
* Ignore all data files, will put them in a different repository/data storage in the future
* Ignore all logs folders
* Update stats analysis file
* Ignore vdiuser on machines
* Fix cryptic error of pickle dump caused by file opened with w+ rather than wb
* Parameterize what is saved, and under which format (JSON Full-Text, or Pickle binary).
* Separate exceptions for NAMO and SNAMO experiments, and fix scenario execution overflow.
* Fix multiprocess execution
* Update rviz file
* Add some logging in tests
* Start cleaning stats utils
* Parameterize min and max scenario for remote execution.
* Keep sim folder, it is actually useful.
* Remove unnecessary simulation subfolder.
* Properly separate simulation history, stats, logs and exceptions into different files.
* Add proper recovery behavior when robot is stuck in an occupied cell though there is no geometrical collision.
* Set the random seed and make it a parameter.
* Fix r_acc_cells definition to properly take into account cases where the robot starts within an obstacle.
* Add precision check to SVG to shapely conversion to make sure there are no duplicate points in geometries.
* Fix forgotten ignored_entities and counter horizon check in transfer path get_conflicts.
* Make stats generation feasible for single scenario.
* Fix counting of total number of goals
* Update requirements
* Add proper robot action space reduction to Stilman Algorithm to achieve actual completeness.
* Add pytrace to requirements
* Make it so prev_list always contains current component, to prevent unneccessary computations.
* Fix Stilman Behavior to better reflect original intended logic by using only the initial connected components, and no longer updating them during planification (which may cause the planning to be infinite in some cases instead of just failing to find a plan.).
* Fix unproper angle_is_close method by removing abs(). Caused transit paths not to have some necessary rotations and thus strange plan executions.
* Refactor test case.
* Fix postponement implementation
* Finally, some decent graphs !
* Slighty refactor stats aggregation to make it more robust to changes
* Slightly filter conflicts
* Complete statistics overhaul.
* Remove GUI for expes
* Add latest stats
* Fix c_1 component for when no c_1, and prepared everything else for night run.
* Add timeout and exception handling
* Properly ignore dynamic obstacles in obstacle choice.
* Fix interblockage caused by always ignoring dynamic obstacles during planning.
* Fix misnamed variable that caused exception.
* Improve debugging a bit, fix basic goal tolerance to be more lax while we wait for a better fix.
* Improve logs, fix ignoring dynamic entities in planning that was not working properly, fix postponement sequence.
* Fix missing parameter
* Save plan history in log report.
* Start refactor for simulation history.
* Changed so much stuff, no time for proper commit message.
* Start implementing new local coordination method.
* Set diff 60 as default action space
* Fix KeyError on Collision detected by simulator before execution.
* Add possibility for holonomic robot (with discretized action set for now), with absolute or not translations.
* Don't change local opening check back to previous AABBTree for the moment.
* Doing everything to get Box2D to work reliably, to no avail...
* Fix class name for iros paper test cases
* Reintegrate CSV collision model so that it is switchable with Box2D
* Kepp GUI by dfault
* Change rviz viz
* Upgrade SVG file Inkscape version.
* Play around sandbox
* Start writing proper tests for collision model.
* Largely rewrite csv-based collision model to be better.
* Make it possible to specify if deg or rad in utils
* Don't say we updated entities that have not changed.
* Properly return parent class returned values
* Clean obstacle object by commenting methods that are deprecated.
* Fix movability deduction function
* Completely rewrite Box2D usage
* Change sandbox experiment to understand what went wrong with Box2D usage.
* Change simulator to use new b2sim collision check API
* Remove deprecated bit in basic actions
* Remove grid update from sense function and have transit paths be verified with b2sim instead. Also fixed start pose/polygon to use action index !
* Redeactivate GUI by degault
* Apply nb of step correction to check_actions function too, to remove unexpected behaviors.
* Make logging optional.
* Remove grid display from default (improves performance).
* Improve binary occupancy grid API to allow reuse in manip_search and to be more trustworthy.
* Fix forgotten grid update in select_connect.
* Simple case finally works !
* Remove GoToPose
* Fixed bugs to the point where I need to remove GoToPose action (which deserved to be deprecated long ago).
* Fixed bugs to the point where simplest case runs without Exceptions, but robot does not reach goal in simulation.
* Fix lingering bugs to the point generating plans works again.
* Add finishing touch to set movability attribute of entities properly.
* Finish rewriting simulation act loop.
* Slight b2sim function call refactor.
* Small 360 angle clamp refactor
* Add method to check many actions for many agents in b2_collision
* Modify rotation action to directly use an angle parameter
* Start exploring a possible more appropriate rewrite of b2sim using Box2D joints, that were avoided until then because of warnings as to their stability.
* Add world step on entity position update to make sure aabbtree is updated.
* Remove goal generation logic from simulator. This is now done in a separate class, which makes much more sense.
* Remove irrelevant code from simulator as long as Wu&Levihn and StandardNavigation are not refactored.
* Fix act method to first check validity of grab and release, properly taking into account simultaneous grabs.
* Merge changes in branch master into experimental b2d branch
* Add minimalist data
* Last minute corrections for exp batch
* Add replan limitation
* Manage the fact that inkscape writes two different elements for paths sometimes.
* Update Rviz config
* Remove unnecessary data
* Create proper scenario generation routine.
* Finish refactoring Stilman behavior to coherently use Box2D collision model. Also updated local opening check to use Box2D aabb tree that is better than custom one.
* Remove deprecated method and fields from TransitPath class
* Make it possible to query b2sim aabb tree
* Refactor find_best_transfer_end_configuration to use Box2D
* Complete teleportation check with ghost
* Clean code where IDE gives warnings
* Clean code where IDE gives warnings
* Finish refactoring plan validity checks with Box2D sim
* Remove deprecated collision_action
* Rewrite transfer path is_valid method to use Box2D.
* Make it possible to activate/deactivate entities in b2sim and grids
* Slightly refactor plan is_valid method to reduce line count (and potential forgotten bugs).
* Rewrite plan is_valid method to use grid for transit paths and Box2D for transfer paths.
* Change action check method to action sequence check
* Refactor Plan, TransitPath and TransferPath is_valid method to reflect changes in collision check model.
* Added b2_sim and relevant occupation grids as Stilman 2005 behavior fields.
* Add capability to return all obstacles in cell to inflated binary occupancy grid
* Deprecate Stilman 2005 Behavior old unit tests
* Move some parameters around
* Big model change to allow update of b2Sim on each sense call
* Add b2Sim to init, start adding it to manip_search and sense methods too
* Rewrite omniscient sensor in a simpler, more efficient way.
* Remove more deprecated code from entity model
* Save a body copy function that could be contributed later to pybox2d
* Fix forgotten argument in b2_collision.py
* Continue cleaning up world.py
* First satisfying version of b2_collision.py
* Remove long deprecated code from entity based representations
* Step 1: Remove old collision checking from Stilman2005Behavior.manip_search_get_neighbors method.
* Change sequence to 50
* Fix logging in simulator to remove newly introduced exception.
* Ignore goals that don't have a geometry definition.
* Add relevant stats graph generation.
* Augment number of used CPUS
* Improve logging
* Add problematic file test
* Add TODO in code of simulator.py
* Fix none goal returned in GoalFailed actions.
* Don't add problems.json files to git
* Fix relative path generation in scenario generation
* Fix scenario generation from logs to use json and not yaml
* Differentiate IOError from ValueError
* Add problems.json file to .gitignore
* Fix little bugs for night-long run
* Make it possible for world to import goals from json
* Remove migration file.
* Migrate world and simulation configs file formats from YAML to JSON.
* Add some extra logs pertaining to the choice of obstacles to consider
* Add needed check to avoid exception in path validity checks
* Thoroughly reinforce logging of robot actions and fix stilman behavior in multi-robots setting to allow our random time draw strategy to work.
* Add extra logging for run method in simulator
* Slightly change simulation with 4 robots
* Modify rviz visualization file to accomodate 4 robots
* Improve logging at simulation-level and fix logging of intermediate world states in SVG with entities traces.
* Finally remove dependency to robot_uid from simulation world display, a long forgotten TODO.
* Applied transform in 4 robots case where the 4th one was not imported correctly because of it.
* Add scenario problem detection in stats analysis
* Add integration tests for 4 robots and new conflict scenarios.
* Add better logging capabilities.
* Add full scenario regeneration code
* Add test file for new SVG import system
* Prevent automatically generated after_the_feast scenarios from being added to repo
* Add 4 robots scenario and second conflict scenario
* Refactor stats analysis, add uncumulated criteria and failed goals criterion
* Improve stats aggregation so that we have figures with distribution, median and mean for each performance criterion.
* Move stats aggregation module to a proper place.
* Change background simulation to kill any remaining processes just in case.
* Force addition to pythonpath
* Force addition to pythonpath
* Plan expe code
* Quick and dirty modification to allow scenario to use same goals
* Final cleanups for Python3.
* Safe load YAML, removes warnings.
* Final cleanups for Python3.
* Fix zip() usage for Python3.
* Fix zip() usage for Python3.
* Fix custom PriorityQueue for Python3 by adding __bool_\_ function definition.
* Fix dict usage for Python3
* Fix clean_attributes function to avoid editing dict being iterated.
* Move ROS colors out of config file to remove ros dep.
* Remove more dependecy to ros
* Remove more dependecy to ros
* Remove more dependecy to ros
* Remove more dependecy to ros
* Fix imports
* Fix test cases to not depend on python launch folder
* Move colors module import to avoid ros dep problem in headless
* Fix forgotten ros dep
* Prevent infinite loops or bugs from crashing the whole simulation.
* Update after the feast complexified case.
* Rewrite sampling methods so that they are easier to understand and fit new BinaryOccupancyGrid API.
* Add conflict test case and temporarily deprecate navigation_only and wu_levihn tests.
* Raise number of goals to 200 for random goal no reset tests
* Update Rviz views
* Ignore RUBE Json dataset meant for tests
* Add a JSON encoder and decoder for Box2D world objects
* Reimplement baseline NAMO algorithm from standard Dijkstra implementation
* Make dijkstra compatible with multiple start configurations
* Filter out other robots from movable obstacles, and manage case when a transfer path is invalidated.
* Fix edge case where no obstacles have been traversed yet
* Change defaults of rviz visualization
* Fix plan validity check so that a path verification will ignore obstacles that need to be moved before its execution and check obstacle start pose with set precision
* Add forgotten opening check when looking for alternative transfer end obstacle poses while planning, and have it use a meaningful start cell ! (the one where the robot is after it has manipulated the obstacle and ready to do a transit path)
* Fix accidentally unindented break statement that caused pre-existing ccs to be redetected as new one
* Add visualization of manip_search close
* Add test case for robot_02 only in after_the_feast_complexified test
* Remove static collision checking at polygon level because its actually slowing down computations instead of accelerating them
* Stop using grid poses as fixed precision poses because it results in incomplete search tree
* Replace minimum_rotated_rectangle calls with manual geometry computation for faster results
* Correctly pass the aabb_tree around in recursive calls to reduce unnecessary computations
* Deprecate NavigationOnlyBehavior until API is stabilized
* Group visualization markers in a common folder for faster display reset
* Fix call to PIL fill procedure so that it does not forget a cell once in a while
* Add debug fill parameter to check discretization cell missing
* Fix Grab action translation vector to be in correct direction
* Quick clean convert_action method
* Fix pose prediction for translation actions : the robot direction angle was not taken into account for transported entity
* Improve manip_search data visualization with proper neighbors and polygons
* Add display function to focused_manip_search
* Start working on proper display function for manip_search
* Rewrite pose_to_arrow function properly
* Remove forgotten duplicate of act function
* Fix grid used for start transit pose validity verification, was not the one with the right radius.
* Fix faulty removal of polygon from the incremental binary occupancy grid
* Fix connectivity grid computation calls
* Rewrite find_best_transfer_end_configuration method for better understanding, bug fixing and performance
* Use min radius for pre-collision check now
* Change default sim to b2
* Fix performance and blending of rch paths
* Add visualization for came_from paths
* Add came_from dict to get_neighbors methods parameters
* Remove custom triangulation module, because was long deprecated
* Remove mplt import to accelerate imports speed when run when no display
* Add OrderedSet data structure
* Improve disassocation from ROS and rewrite RCH display function
* Add conversion function for single grid_cell
* Group color-related functions in colors.py
* Pass along list of traversed obstacles ids for fast display
* Fix close_set reading in A*
* Manage edge in old collision detection method based on convex enveloppes
* Clean up SVG reading to polygon to remove duplicate points
* Integrate Box2D collision detection in Stilman manip search
* Finish up first good implementation of Box2D-based collision detection
* Continue experiment with Box2D
* Separate computation of directed translation vector for Translation action
* First working Box2D-based simulation working
* Try box2d collision joining of bodies as fixtures in 1: failed
* Try to use box2d for manip search simulations
* Add utils to convert concave polygons to convex polygons if they are not so
* Comment deprecated code for polygon trace in SVG files while its not reimplemented correctly
* Add open_queue to get_neighbor functions parameters
* Rename src folder as snamosim to make proper pip packaging possible. Also fix rasterization.
* Add dependencies to setup.py
* Add necessary python files
* Merge branch 'master' of https://gitlab.inria.fr/brenault/s-namo-sim-private
* src/requirements.txt
* Remove deprecated files and imports
* Various fixes to stilman behavior so that it fits
* Fix grid parameters computation so that grid_pose is properly computed and more relevant data is returned.
* Continue cleanup in world defition
* Experimenting ways to have accurate AND fast rasterization
* Fix API call to BinaryOccupancyGrid
* Start writing tests to check rasterization, since it revealed to not be so obvious.
* Temporarily deprecate WuLevihn behavior because APIs changed and it need updates
* Fix ros-conversion for world using new costmap
* Fix a star call to new API in navigation only behavior
* Add possibility for A* to take multiple starts
* Finish all refactors using new graph search methods and removed deprecated code
* Add tests for A*, fix priority queue implementation
* Slightly reorganize simulator.py
* Completely change API between simulator.py and behaviors so that behaviors can be properly synchronized and collisions between actions properly detected.
* Advance implementation of proper deterministic multi-robot action execution model
* Light refactoring of simulator.py, start separating sense-think-act into three loops, to go to deterministic behavior.
* Light refactoring of new_stilman_2005_behavior.py and fixed Transit+Transfer paths definitions
* Remove intersection from collision_data in collision.py
* Mild refactoring and advancing new transfer path definition
* Refactor Configuration for more general use and remove remaining warnings.
* Remove outdated code in new_stilman_behavior, fixed warnings and made sure transit paths start and end poses were in free space.
* Remove LinearMovement from collision.py because the assumption it relied on revealed to be wrong
* Add to utils.py the functions: get_neighbors_no_checks, get_set_neighbors_no_checks and sample_poses_at_middle_of_inflated_sides
* Add data of  debug test cases of after_the_feast scenario
* Add specific cases checks in find_circle_terms, added are_points_on_opposite_sides and sum_of_euclidean_distances functions.
* Try new stuff in sandbox
* Add debug scenario after 16 goals no reset in after_the_feast test suite
* Catch all think exceptions and fully print them out before trying again.
* Reset ros publisher config to the right default value
* Fix rviz display of paths
* Continue improving stilman_2005_behavior
* Fix Generic A Star return order
* Checkpoint
* Checkpoint
* Checkpoint
* Checkpoint
* Checkpoint
* Big update, fixed big performance issues and a wide array of bugs
* More updates
* Big update
* Forgotten commit, too many things changed.
* Big update for IROS 2021 submission
* Enormous update, too many things changed and I need a snapshot of current developments
* Fix test for TwoRoomsCorridor case
* Add test for chen difficult problem
* Add methods to compute transferred obstacles to plan.py
* Init report classes
* Update rviz config
* Add chen difficult problem
* Update tests with latest interface of connected components grid
* Fix bug caused by last_action_success=None in Wu Levihn Behavior
* Fix local opening detection exception when intersection geometries are Multipolygons instead of Polygons
* New rviz config
* Remove no longer needed methods from stilman 2005 behavior
* Add inherited new parameter to Robot and Obstacle classes
* Fix rotation bug with omniscient sensor
* Fix None return to None, None tuple
* Add Tests for Moghaddam
* Add Mogghadam simulation data files
* Slightly refactor to remove unneeded connected components grid computations and change basic_rot_force to basic_rot_moment constants.
* Change actions to proper objects instead of generated functions and use that to implement proper collision detection and improved configuration check.
* Remove TODO at top of Stilman 2005 Behavior class
* Add tests for second scenario (big crossing).
* Add IntersectionError exception forgottent at last commit
* Implement better A Posteriori (discrete) collision detection when moving obstacle.
* Fix improper rotation detection in g_fov_sensor
* Fix exception when trying to polygonize a LineString or Point
* Update rviz visualization config
* Update world and simulation definitions
* Add support for networkx and experimental support for stinger libraries for computation of connected components.
* Improve and add unit tests for stilman 2005 main methods (manip_search, rch and select_connect)
* Add stilman 2005 behavior to two rooms corridor test.
* Add rotation support in simulator.py
* Add support for stilman 2005 behavior in simulator.py
* Add omniscient sensor for stilman 2005 algorithm
* Add connected components topic and flashy_red color in ros publisher config
* Reorganize ros_publisher code by region and add stilman 2005 specific functions
* Add method to initialize grid_map and update grid_map conversion function.
* Update world conversion to rviz markers for sensors
* Fix utility functions and classes of stilman 2005 behavior
* Fix manip_search subfunction of stilman 2005 behavior
* Complete select_connect method for stilman 2005 behavior
* Rewrite rch subfunction for stilman 2005 behavior
* Implement think method for Stilman 2005 behavior
* Add passing of rot_center parameter in rotate_entity procedure to allow proper rotations of obstacle
* Generize sensors structure
* Implemented (not functional yet) incremental update of connected components grid
* Update binary inflated occupancy grid to save sets of freed and invaded cells as they are updated
* Remove use of properties for grid because causes strange behavior during debug
* Improve gitignore to ignore all venvs
* Update rviz config
* Update worlds data for 01_two_rooms_corridor scenario and simulation data for stilman 2005 behavior
* Decide on a proper first rotation energy cost approximation and fix manip_search consequently to account for float precision variations
* Get back to using A* rather than multi-goal A* in stilman 2005
* Improve a bit the Rviz display's organization
* Fix default rotation angles
* Add a cleanup_robot_sim method and use it
* Fix use of check new opening in wu levihn 2014
* Refactor pose rounding into its own function
* Update rviz config accordingly
* Refactor ros_publisher so that no message is computed when the RVIZ tickboxes are not ticked, move parameters into a separate file for easier access and modification, move conversion functions to specific file to improve readability and finally, fix the Segfault that happened when using GridCells by no longer using them and using CUBE_LIST-type Markers instead.
* Fix connected components update test so that we care more about the connected components cells sets than the grids
* Fixed connected components grid so that all tests would pass
* Fix problem when more cells are freed than created. Still have to rewrite the test a bit so that if components are right, ids don't really matter
* Add implementation of incremental grid updates for connectivity grid with tests
* Move CellHeapNode in self-contained class
* Fix _rch and _select_connect methods to fit with the changes made while modifying manip_search
* Add append function to Plan Class to be used by Stilman2005Behavior
* Fix rch method in stilman 2005 behavior according to the changes made in init
* Refactor world to send back full grid object rather than underlying numpy matrix
* Refactor world according to change in robot sensors definition
* Add mostly complete test of manip_search function
* Finally has a broadly acceptable manip_search method for stilman 2005 behavior
* Fix grid call in navigation only behavior after grid refactor
* Add default float precision constants in baseline behavior
* Take local new opening detection outside of Wu And Levihn, refactor and fix it.
* Add publish method for blocking areas and diameter inflated polygons to easily debug new local opening detection
* Fix sensor publishing in ros_publisher after refactor
* Fix grid use in function conversion of ros_publisher after grid object refactor
* Add automatic color conversion in ros_publisher
* Add light_copy for Robot class
* Fix sensor rotation bug in Robot class
* Refactor ros_publisher use in obstacle to remove warning
* Implement light_copy for Obstacle
* Add intersects/discrete_interescts and light_copy functions to entity class
* Fix fov rotation bug, entity side
* Fix fov rotation bug
* Add forgotten check for fov sensors for robot uid
* Fix faulty constants in utils
* Add possibility to break on first goal reached for multigoal a star
* Add conversion to real path function for multigoal a star
* Update rviz config file
* Update rviz config file
* Change actions_branch_to_path to static method
* Temporarily switch to min_inflation_radius to get manip poses
* Refactor Stilman Behavior so that it can inherit from BaselineBehavior
* Correctly call manip_search method in tes
* Fix Simulator call to Stilman2005Behavior with right parameters
* Fix multigoal astar (gscore init + check in to_evaluate_set + check in goal_s)
* Start testing stilman 2005 behavior, starting with manip_search method
* Get components as property
* Fix World get_connected_components_grid method and add get_connected_components method
* Add stilman 2005 behavior for two rooms corrider test data file
* Add a bit of doc to CCG Class init_grid method
* Rewrite procedure to get distance between robot center and front side
* Fix action computation procedure in Robot class (threw exception because of bad type)
* Add opening detection method described in article to stilman 2005 behavior
* Clean up abusive of dd to pass discretization data
* Fix connected components grid and add associated unit tests
* Add proper action simulation and registration to stilman 2005 behavior
* Create utils method to check if any cell in a cell set collides in a given grid
* Add proper usage of multi goal a star in stilman 2005behavior
* Fix multi-goal a star open heap display method
* Add proper usage of manipulation poses in stilman 2005 behavior
* Add proper method for computing the center to border distance at which the robot's front would touch an obstacle.
* Separate _compute_possible_actions into a manipulation poses generation procedure and an action generation procedure.Also added outline for different manipultion poses generation procedure.
* Add direction_from_yaw and interval check functions in utils
* Fix multi-goal a-star heap use, and clean up its code a bit
* Start writing some tests files in prevision of stilman 2005 behavior
* Add a report system so that we obtain the necessary statistics for comparing approaches
* Start implementation of simulation report
* Add goal not None check in ros_publisher
* Change goal poses to tuples so that they can be hashed and used as dictionnary keys
* Add bit of doc to wu/levihn
* Got Wu/Levihn back working again with significant performance improvement as to grid generation
* Refactor wu/levihn behavior
* Refactor calls to utils module
* Clean up nav only behavior
* Remove use of social layers in basic tests of Wu/Levihn
* Change simulator feedback from booleans to actual objects that contain useful information as to what has just been done
* Refactor wu/levihn parameters
* Add notes in social costmap code for future reference
* Add more integration tests for the two rooms corridor scenario
* Add new parameters to nav behaviors in simulator.py
* Display binary inflated grid rather than probabilist one
* Update behaviors according to refactor (not finished for wu/levihn)
* Fix misplaced call to RosPublisher that caused an infinite loop during imports
* Add two rooms corridor world but without obstacles
* Update RVIZ conf
* Add simulation data for basic use cases with Wu/Levihn and NavOnly
* Tremendous refactor, nothing is as it was before.
* Fix typo in Stilman 2005 behavior
* Add some cases to simulator.py
* Continue big refactor of alternate world representations
* Remove useless ros_node.py file
* Create separate files and classes for alternative world representations
* Update Rviz config
* Remove old worlds and add new ones since format update
* Fix rasterization bug that caused creation of wrong costmaps
* Fix error message in rviz caused by invalid quaternion for static transform
* Add function to compute minimal robot inflation radius online
* Fix division by inf in edge case of a polygon with two almost exactly same points
* Continue implementing manip_search method in stilman_2005 behavior (still not usable)
* WIP: Implementation of critical manip_search method to stilman_2005 behavior (not usable yet)
* Add constants relative to standard robot action spaces in discrete environments
* Change path to SVG in YAML file for moghaddam_planning_2016_benchmark/01 to better reflect possibilities
* Add commodity tmux launch file at project root for fast and automated setup of dev environment
* Add new behabior corresponding to 2005 Stilman+Kuffner paper
* Reorganize data files and formats to use SVG for geometry + YAML for semantics
* Transfer some commonly used methods and constants into utils (is_in_matrix, neighborhoods, ...)
* Rename entity.py to thing.py
* Add triangulation class need to fix the display problem with concave obstacles
* Refactor publishing to improve performance and genericity, still a way to go. Fix display problem with concave obstacles
* Update robot behaviors and Simulator class to match with new world model
* Refactored world model to add import from svg feature, and add social costmap model beta version
* Refactor physical object classes in preparation of SVG import
* Update ptpython requirement to fix trouble caused by latest version
* Update rviz file to visualize social costmap
* Forgot to push latest devs...
* Initial commit
* Initial commit
* Contributors: BROWN David, Benoit Renault, David Brown, RENAULT Benoit, Xia0ben, brenault, semantic-release
