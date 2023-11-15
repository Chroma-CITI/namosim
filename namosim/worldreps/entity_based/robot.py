import copy

from namosim.utils import utils
from namosim.worldreps.entity_based.entity import Entity, Style


class Robot(Entity):
    def __init__(
        self,
        name: str,
        full_geometry_acquired,
        polygon,
        pose,
        sensors,
        push_only_list,
        force_pushes_only,
        movable_whitelist,
        style: Style,
        movability="unknown",
        uid=0,
    ):
        polygon = polygon
        Entity.__init__(
            self,
            name=name,
            polygon=polygon,
            pose=pose,
            full_geometry_acquired=full_geometry_acquired,
            movability=movability,
            uid=uid,
            style=style,
            type_="robot",
        )

        self.sensors = sensors
        for sensor in sensors:
            sensor.parent_uid = self.uid

        self.push_only_list = push_only_list
        self.force_pushes_only = force_pushes_only
        self.movable_whitelist = movable_whitelist
        self.min_inflation_radius = self.compute_inflation_radius()

    def update_world_from_sensors(self, reference_world, target_world):
        added_uids, updated_uids, removed_uids = set(), set(), set()

        for sensor in self.sensors:
            s_uids_to_add, s_uids_to_update, s_uids_to_remove = sensor.update_from_fov(
                reference_world, target_world
            )

            # Might need a better update policy if sensors disagree about what happened, but irrelevant for now
            added_uids.update(s_uids_to_add)
            updated_uids.update(s_uids_to_update)
            removed_uids.update(s_uids_to_remove)

        return added_uids, updated_uids, removed_uids

    def deduce_movability(self, obstacle_type):
        if obstacle_type == "unknown" or obstacle_type == "robot":
            return "unknown"
        elif obstacle_type in self.movable_whitelist:
            return "movable"
        else:
            return "unmovable"

    def deduce_push_only(self, obstacle_type):
        if self.force_pushes_only or obstacle_type in self.push_only_list:
            return True
        else:
            return False

    def compute_inflation_radius(self):
        return utils.get_circumscribed_radius(self.polygon)

    def light_copy(self):
        return Robot(
            name=self.name,
            polygon=copy.deepcopy(self.polygon),
            pose=self.pose,
            full_geometry_acquired=self.full_geometry_acquired,
            sensors=copy.deepcopy(self.sensors),
            push_only_list=copy.copy(self.push_only_list),
            force_pushes_only=self.force_pushes_only,
            movable_whitelist=copy.copy(self.movable_whitelist),
            uid=self.uid,
            style=self.style,
        )

    def to_json(self):
        json_data = Entity.to_json(self)
        json_data["geometry"]["orientation_id"] = self.name + "_dir"
        json_data["movable_whitelist"] = self.movable_whitelist
        json_data["push_only_list"] = self.push_only_list
        json_data["force_pushes_only"] = self.force_pushes_only
        json_data["sensors"] = []
        for sensor in self.sensors:
            json_data["sensors"].append(sensor.to_json())
        return json_data
