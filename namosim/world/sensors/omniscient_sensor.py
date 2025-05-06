import copy

import namosim.world.world as world


class OmniscientSensor:
    def __init__(self):
        self.parent_uid: str | None = None

    def update_from_fov(
        self, reference_world: "world.World", target_world: "world.World"
    ):
        # Add
        uids_to_add = set(reference_world.dynamic_entities.keys()).difference(
            target_world.dynamic_entities.keys()
        )
        for uid in uids_to_add:
            target_world.add_entity(reference_world.dynamic_entities[uid].copy())

        # Update
        uids_to_potentially_update = set(
            reference_world.dynamic_entities.keys()
        ).intersection(target_world.dynamic_entities.keys())
        uids_to_update: set[str] = set()
        for uid in uids_to_potentially_update:
            ref_entity = reference_world.dynamic_entities[uid]
            target_entity = target_world.dynamic_entities[uid]
            if target_entity.pose != ref_entity.pose:
                target_entity.pose = ref_entity.pose
                target_entity.polygon = ref_entity.polygon
                uids_to_update.add(uid)

        # Remove
        uids_to_remove = set(target_world.dynamic_entities.keys()).difference(
            reference_world.dynamic_entities.keys()
        )
        for uid in uids_to_remove:
            target_world.remove_entity(uid)

        # Copy all grab data from reference world
        target_world.entity_to_agent = copy.deepcopy(reference_world.entity_to_agent)

        return uids_to_add, uids_to_update, uids_to_remove
