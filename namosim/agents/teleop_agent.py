import typing as t

import numpy as np
from shapely import Polygon

import namosim.display.ros2_publisher as rp
import namosim.navigation.basic_actions as ba
import namosim.world.world as w
from namosim.agents.agent import Agent, ThinkResult
from namosim.data_models import UID, PoseModel
from namosim.input import Input
from namosim.utils import utils
from namosim.world.entity import Style
from namosim.world.sensors.omniscient_sensor import OmniscientSensor


class TeleopAgent(Agent):
    def __init__(
        self,
        *,
        navigation_goals: t.List[PoseModel],
        logs_dir: str,
        name: str,
        full_geometry_acquired: bool,
        polygon: Polygon,
        pose: PoseModel,
        sensors: t.List[OmniscientSensor],
        style: Style,
        logger: utils.CustomLogger,
        cell_size: float,
        uid: UID = 0,
    ):
        Agent.__init__(
            self,
            name=name,
            navigation_goals=navigation_goals,
            behavior_type="telop_behavior",
            logs_dir=logs_dir,
            full_geometry_acquired=full_geometry_acquired,
            polygon=polygon,
            pose=pose,
            sensors=sensors,  # type: ignore
            style=style,
            logger=logger,
            cell_size=cell_size,
            uid=uid,
        )
        self.neighborhood = utils.CHESSBOARD_NEIGHBORHOOD
        self.robot_max_inflation_radius = utils.get_circumscribed_radius(self.polygon)

    def init(self, world: "w.World"):
        super().init(world)

    def think(
        self, ros_publisher: "rp.RosPublisher", input: t.Optional[Input] = None
    ) -> ThinkResult:
        next_action = ba.Wait()

        if input is None:
            next_action = ba.Wait()
        elif input.key_pressed == "Up":
            next_action = ba.Advance(distance=self.cell_size)
        elif input.key_pressed == "Down":
            next_action = ba.Advance(distance=-self.cell_size)
        elif input.key_pressed == "Left":
            next_action = ba.Rotation(30)
            input.clear()
        elif input.key_pressed == "Right":
            next_action = ba.Rotation(-30)
            input.clear()
        elif input.key_pressed == "g":
            next_action = self._grab()
            input.clear()
        elif input.key_pressed == "r":
            next_action = self._release()
            input.clear()

        return ThinkResult(
            next_action=next_action,
            goal_pose=None,
            did_replan=False,
            robot_name=self.name,
        )

    def _grab(self) -> ba.Grab | None:
        movables = self.world.get_movable_obstacles()
        for m in movables:
            d = m.polygon.distance(self.polygon)

            if d > self.circumscribed_radius:
                continue

            angle = utils.get_angle_to_turn(self.pose, m.pose)
            if np.abs(angle) > 10:
                continue

            return ba.Grab(d, m.uid)

    def _release(self) -> ba.Release | None:
        if self.world.is_holding_obstacle(self.uid):
            return ba.Release(
                -self.grab_and_release_distance,
                self.world.entity_to_agent.inverse[self.uid],
            )
