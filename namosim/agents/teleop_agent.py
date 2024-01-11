import typing as t

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
        push_only_list: t.List[str],
        force_pushes_only: bool,
        movable_whitelist: t.List[str],
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
            push_only_list=push_only_list,
            force_pushes_only=force_pushes_only,
            movable_whitelist=movable_whitelist,
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
            next_action = ba.Translation((self.cell_size, 0))
        elif input.key_pressed == "Down":
            next_action = ba.Translation((-self.cell_size, 0))
        elif input.key_pressed == "Left":
            next_action = ba.Rotation(-30)
        elif input.key_pressed == "Right":
            next_action = ba.Rotation(30)

        return ThinkResult(
            next_action=next_action,
            did_replan=False,
            robot_name=self.name,
            has_conflicts=False,
        )
