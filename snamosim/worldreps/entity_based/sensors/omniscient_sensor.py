class OmniscientSensor:
    def __init__(self):
        self.parent_uid = None

    def update_from_fov(self, reference_world, target_world):
        for ref_uid, ref_entity in reference_world.entities.items():
            if ref_uid in target_world.entities:
                # Update
                target_entity = target_world.entities[ref_uid]
                if target_entity.pose != ref_entity.pose:
                    target_entity.pose = ref_entity.pose
                    target_entity.polygon = ref_entity.polygon
            else:
                # Add
                target_world.add_entity(ref_entity.light_copy())
        # Remove
        uids_to_remove = set(target_world.entities.keys()).difference(reference_world.entities.keys())
        target_world.remove_entities(uids_to_remove)

    def to_json(self):
        return {"type": "omniscient"}
