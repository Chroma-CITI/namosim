import copy


class OmniscientSensor:
    def __init__(self):
        self.parent_uid = None

    def update_from_fov(self, reference_world, target_world):
        # Add
        uids_to_add = set(reference_world.entities.keys()).difference(target_world.entities.keys())
        for uid in uids_to_add:
            target_world.add_entity(reference_world.entities[uid].light_copy())

        # Update
        uids_to_potentially_update = set(reference_world.entities.keys()).intersection(target_world.entities.keys())
        uids_to_update = set()
        for uid in uids_to_potentially_update :
            ref_entity = reference_world.entities[uid]
            target_entity = target_world.entities[uid]
            if target_entity.pose != ref_entity.pose:
                target_entity.pose = ref_entity.pose
                target_entity.polygon = ref_entity.polygon
                uids_to_update.add(uid)

        # Remove
        uids_to_remove = set(target_world.entities.keys()).difference(reference_world.entities.keys())
        for uid in uids_to_remove:
            target_world.remove_entity(uid)

        # Copy all grab data from reference world
        target_world.entity_to_agent = copy.deepcopy(reference_world.entity_to_agent)

        return uids_to_add, uids_to_update, uids_to_remove

    def to_json(self):
        return {"type": "omniscient"}
