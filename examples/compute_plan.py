from namosim.navigation.action_result import ActionSuccess
from namosim.world.world import World


world = World.load_from_svg(
    "tests/scenarios/minimal_stilman_2005.svg",
)
agent = world.agents['robot_0']
agent.sense(ref_world=world, last_action_result=ActionSuccess(), step_count=0)
think_result = agent.think()
assert think_result.plan is not None
assert len(think_result.plan.paths) == 3
assert think_result.plan.paths[1].is_transfer