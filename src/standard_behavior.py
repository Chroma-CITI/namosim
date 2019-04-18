import a_star
from path import Path
from plan import Plan
import numpy as np


class StandardBehavior:
    def __init__(self, simulator, initial_world, robot_uid):
        self.simulator = simulator
        self.robot_uid = robot_uid
        self.world = initial_world

    def execute(self, q_init, q_goal, rp):
        q_r = q_init
        p_opt = Plan([Path([])])

        while not all(np.isclose(q_r, q_goal, rtol=0.001)):
            self.simulator.update_robot_knowledge(self.world)
            q_r = self.world.entities[self.robot_uid].pose

            if not p_opt.is_valid(self.world):
                p_opt = Plan([Path(a_star.a_star_real_path(
                    self.world.get_grid(), q_r, q_goal, self.world.dd, rp))])

            if p_opt.is_not_empty():
                next_step = p_opt.pop_next_step()
                self.simulator.try_exe_next_step(self.robot_uid, next_step)
            elif p_opt.has_infinite_cost():
                return False
        return True
